import torch
import numpy as np
import os
import sys
from PIL import Image
from typing import List, Tuple, Union
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from einops import rearrange
import tempfile
import rembg
from torchvision.transforms import v2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add InstantMesh submodule to path so 'src' can be imported
INSTANT_MESH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'InstantMesh'))
if INSTANT_MESH_ROOT not in sys.path:
    sys.path.append(INSTANT_MESH_ROOT)

from text_to_3d.text_to_3d import TextTo3D

# Try importing from src (InstantMesh submodule)
try:
    from src.utils.train_util import instantiate_from_config
    from src.utils.camera_util import (
        FOV_to_intrinsics, 
        get_zero123plus_input_cameras,
        get_circular_camera_poses
    )
    from src.utils.mesh_util import save_obj_with_mtl
    from src.utils.infer_util import remove_background, resize_foreground
except ImportError as e:
    print(f"Error importing from InstantMesh submodule: {e}")
    print(f"sys.path: {sys.path}")
    raise e

class InstantMesh(TextTo3D):
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.zero123plus_pipe = None
        self.instant_mesh_model = None
        self.infer_config = None
        self.is_flexicubes = False

    def init_model(self):
        if self.zero123plus_pipe is None:
            print("Loading Zero123++ model...")
            # We use the pipeline logic from the draft/submodule
            custom_pipeline_path = os.path.join(INSTANT_MESH_ROOT, "zero123plus")
            
            self.zero123plus_pipe = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2", 
                custom_pipeline=custom_pipeline_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.zero123plus_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.zero123plus_pipe.scheduler.config, 
                timestep_spacing='trailing'
            )
            self.zero123plus_pipe.to(self.device)

        if self.instant_mesh_model is None:
            print("Loading InstantMesh model...")
            
            config_path = os.path.join(INSTANT_MESH_ROOT, "configs", "instant-mesh-base.yaml")
            
            if not os.path.exists(config_path):
                 raise FileNotFoundError(f"Config not found at {config_path}")

            config = OmegaConf.load(config_path)
            model_config = config.model_config
            self.infer_config = config.infer_config
            
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
            
            self.instant_mesh_model = instantiate_from_config(model_config)
            
            state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
            state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
            self.instant_mesh_model.load_state_dict(state_dict, strict=True)
            self.instant_mesh_model.to(self.device)
            
            # Init flexicubes
            self.is_flexicubes = os.path.basename(config_path).startswith("instant-mesh")
            if self.is_flexicubes:
                self.instant_mesh_model.init_flexicubes_geometry(self.device, fovy=30.0)
            
            self.instant_mesh_model.eval()

    def preprocess(self, input_image, do_remove_background):
        rembg_session = rembg.new_session() if do_remove_background else None
        if do_remove_background:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)
        return input_image

    def generate_mvs(self, input_image, sample_steps=75, sample_seed=42):
        generator = torch.Generator(device=self.device).manual_seed(sample_seed)
        z123_image = self.zero123plus_pipe(
            input_image,
            num_inference_steps=sample_steps,
            generator=generator,
        ).images[0]
        return z123_image

    def convert_image_to_3d(self, image: Union[Image.Image, str], output_dir: str, name_prefix: str = "output") -> str:
        """
        Convert an image to a 3D mesh.
        
        Args:
            image: PIL Image or path to image file.
            output_dir: Directory to save results.
            name_prefix: Prefix for output filenames.
            
        Returns:
            Path to the generated .obj file.
        """
        self.init_model()
        
        if isinstance(image, str):
            image = Image.open(image)
            
        # Debug directory
        debug_dir = os.path.join("results", "instant_mesh_stages")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. Preprocess Image
        print("Preprocessing image...")
        processed_image = self.preprocess(image, do_remove_background=True)

        # Save Preprocessed Image
        processed_image.save(os.path.join(debug_dir, f"{name_prefix}_2_preprocessed.png"))
        
        
        # 2. Generate Multi-View Images
        print("Generating multi-view images...")
        mv_images = self.generate_mvs(processed_image)

        # Save Multi-view Images
        mv_images.save(os.path.join(debug_dir, f"{name_prefix}_3_multiview.png"))
        
        
        # 3. Reconstruct 3D Mesh
        print("Reconstructing 3D mesh...")
        
        # Prepare input for InstantMesh
        mv_images_np = np.asarray(mv_images, dtype=np.float32) / 255.0
        mv_images_tensor = torch.from_numpy(mv_images_np).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        mv_images_tensor = rearrange(mv_images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)    # (6, 3, 320, 320)
        
        mv_images_tensor = mv_images_tensor.unsqueeze(0).to(self.device)
        mv_images_tensor = v2.functional.resize(mv_images_tensor, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)
        
        os.makedirs(output_dir, exist_ok=True)
        mesh_fpath = os.path.join(output_dir, f"{name_prefix}.obj")
        
        with torch.no_grad():
            planes = self.instant_mesh_model.forward_planes(mv_images_tensor, input_cameras)
            mesh_out = self.instant_mesh_model.extract_mesh(planes, use_texture_map=True, **self.infer_config)
            
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_fpath,
            )
            
        print(f"Mesh saved to {mesh_fpath}")
        return mesh_fpath

    def convert_text_to_3d(self, text: str, output_dir: str) -> str:
        raise NotImplementedError("This method is deprecated. Use convert_image_to_3d instead.")

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
         raise NotImplementedError("This method is deprecated. Use convert_image_to_3d iteratively.")


