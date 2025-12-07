# sd_generate.py
from diffusers import StableDiffusionPipeline
import torch


_PIPE = None


def generate_image_with_stable_diffusion(prompt: str,
                                         model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                                         device: str = "cuda"):
    global _PIPE
    if _PIPE is None:
        _PIPE = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    image = _PIPE(prompt).images[0]
    return image


def unload_sd_pipeline():
    """Move the global SD pipeline off GPU and drop it."""
    global _PIPE
    if _PIPE is not None:
        try:
            _PIPE.to("cpu")
        except Exception:
            pass
        del _PIPE
        _PIPE = None

# instantmesh_mv.py
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import torch
from einops import rearrange
import numpy as np
from PIL import Image
import rembg
from instantmesh.src.utils.infer_util import remove_background, resize_foreground

def build_zero123plus_pipeline(device="cuda"):
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline="instantmesh/zero123plus", torch_dtype=torch.float16
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    return pipeline.to(device)

def preprocess(input_image, do_remove_background=True):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image

def generate_mvs(pipeline, input_image, sample_steps=75, sample_seed=42, device="cuda"):
    torch.manual_seed(sample_seed)
    generator = torch.Generator(device=device)
    z123_image = pipeline(input_image, num_inference_steps=sample_steps, generator=generator).images[0]
    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image.copy())
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())
    return z123_image, show_image

import os
import tempfile
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from einops import rearrange

from instantmesh.src.utils.train_util import instantiate_from_config
from instantmesh.src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from instantmesh.src.utils.mesh_util import save_obj, save_obj_with_mtl


# ---------- Paths ----------
# Compute project root = parent of this 'modules' directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "instantmesh", "configs", "instant-mesh-base.yaml")


def _patch_targets_to_vendored(obj, prefix="instantmesh."):
    """
    Recursively walk a dict/list and rewrite any 'target' value that
    starts with 'src.' to 'instantmesh.src.' so instantiate_from_config
    can import from the vendored repo.

    Works on plain Python containers (dict/list/str/…),
    so call it on OmegaConf.to_container(...) results.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "target" and isinstance(v, str) and v.startswith("src."):
                out[k] = prefix + v  # e.g., 'src.models.Foo' -> 'instantmesh.src.models.Foo'
            else:
                out[k] = _patch_targets_to_vendored(v, prefix=prefix)
        return out
    elif isinstance(obj, list):
        return [_patch_targets_to_vendored(x, prefix=prefix) for x in obj]
    else:
        return obj


# ---------- Public API ----------
def load_instantmesh(config_path: str | None = None, device: str = "cuda"):
    """
    Load InstantMesh model + infer_config from the vendored repo.
    Returns: (model, is_flexicubes, infer_config)
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG

    # Load YAML
    cfg = OmegaConf.load(config_path)
    # Convert to plain containers and patch 'target:' strings
    model_config = OmegaConf.to_container(cfg.model_config, resolve=True)
    model_config = _patch_targets_to_vendored(model_config, prefix="instantmesh.")

    infer_config = OmegaConf.to_container(cfg.infer_config, resolve=True)

    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_base.ckpt",
        repo_type="model",
    )
    # Instantiate model
    model = instantiate_from_config(model_config)
    # Load weights (strip 'lrm_generator.' and drop 'source_camera' keys)
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k[14:]: v
        for k, v in state_dict.items()
        if k.startswith("lrm_generator.") and "source_camera" not in k
    }
    model.load_state_dict(state_dict, strict=True)

    config_name = os.path.basename(config_path).replace(".yaml", "")
    is_flexicubes = config_name.startswith("instant-mesh")
    if is_flexicubes:
        model.init_flexicubes_geometry(device, fovy=30.0)

    model = model.to(device).eval()
    return model, is_flexicubes, infer_config


def images_to_video(images: torch.Tensor, output_path: str, fps: int = 30):
    """
    images: (T, C, H, W) in [0,1]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec="h264")


def get_render_cameras(
    batch_size: int = 1,
    M: int = 120,
    radius: float = 2.5,
    elevation: float = 10.0,
    is_flexicubes: bool = False,
):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0)
            .unsqueeze(0)
            .repeat(M, 1, 1)
            .float()
            .flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def make_mesh(mesh_fpath: str, planes: torch.Tensor, model, infer_config):
    """
    Extract a textured mesh to 'mesh_fpath' (.obj). MTL and texture are written next to it.
    Returns mesh_fpath.
    """
    with torch.no_grad():
        mesh_out = model.extract_mesh(
            planes, use_texture_map=True, **infer_config
        )
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out

        save_obj_with_mtl(
            vertices.data.cpu().numpy(),
            uvs.data.cpu().numpy(),
            faces.data.cpu().numpy(),
            mesh_tex_idx.data.cpu().numpy(),
            tex_map.permute(1, 2, 0).data.cpu().numpy(),
            mesh_fpath,
        )
        print(f"Mesh with texmap saved to {mesh_fpath}")
    return mesh_fpath


def make3d(
    mv_grid_image: Image.Image | np.ndarray,
    model,
    is_flexicubes: bool,
    device: str = "cuda",
    infer_config=None,
    output_dir: str | None = None,
    render_size: int = 384,
    M: int = 120,
    radius: float = 4.5,
    elevation: float = 20.0,
):
    """
    Convert Zero123++ multi-view grid into a textured OBJ.

    Returns:
        obj_path, mtl_path, texture_path
    """
    if output_dir is None:
        output_dir = "/content/tmp"
    os.makedirs(output_dir, exist_ok=True)
    tempfile.tempdir = output_dir

    # --- Prepare input views (expects 3x2 tiling like your notebook) ---
    images = np.asarray(mv_grid_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)     # (6, 3, 320, 320)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    # Cameras
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(
        batch_size=1, M=M, radius=radius, elevation=elevation, is_flexicubes=is_flexicubes
    ).to(device)

    # Filenames in output_dir
    obj_path = os.path.join(output_dir, "output_model.obj")

    # --- Forward planes (no turntable video anymore) ---
    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)

    # --- Mesh extraction (OBJ+MTL+texture) ---
    obj_path = make_mesh(obj_path, planes, model, infer_config)

    # Infer companion file paths
    mtl_path = obj_path.replace(".obj", ".mtl")
    texture_path = obj_path.replace(".obj", ".png")

    return obj_path, mtl_path, texture_path