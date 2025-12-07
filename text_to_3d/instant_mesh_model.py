import torch
import numpy as np
import os
import sys
from PIL import Image
from typing import List
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from einops import rearrange
import tempfile
import rembg

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_to_3d.text_to_3d import TextTo3D
from text_to_3d.instant_mesh.src.utils.infer_util import remove_background, resize_foreground
from text_to_3d.instant_mesh.src.utils.train_util import instantiate_from_config
from text_to_3d.instant_mesh.src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses
)
from text_to_3d.instant_mesh.src.utils.mesh_util import save_obj_with_mtl

class InstantMesh(TextTo3D):
    def __init__(self, device: str = "cuda", guidance_scale: float = 7.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.guidance_scale = guidance_scale
        
        self.txt2img_pipe = None
        self.zero123plus_pipe = None
        self.instant_mesh_model = None
        self.infer_config = None

    def init_model(self):
        if self.txt2img_pipe is None:
            print("Loading Stable Diffusion model...")
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.txt2img_pipe.to(self.device)

        if self.zero123plus_pipe is None:
            print("Loading Zero123++ model...")
            # Use the custom pipeline from the folder as requested/implied by the notebook structure
            # But the notebook uses "sudo-ai/zero123plus-v1.2" with custom_pipeline="zero123plus"
            # The user said "I have included the custom pipeline of Zero123plus ... in the folder text_to_3d/zero123plus"
            # So we should point custom_pipeline to that folder.
            custom_pipeline_path = os.path.join(os.path.dirname(__file__), "instant_mesh", "zero123plus")
            
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

            base_path = os.path.dirname(__file__) 
            instant_mesh_root = os.path.join(base_path, "instant_mesh")
            if instant_mesh_root not in sys.path:
                sys.path.append(instant_mesh_root)
            
            # User confirmed configs/instant-mesh-base.yaml exists in the repo
            config_path = os.path.join(instant_mesh_root, "configs", "instant-mesh-base.yaml")
            
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

    def convert_text_to_3d(self, text: str, output_dir: str) -> str:
        self.init_model()
        
        # 1. Text to Image
        print(f"Generating image for: {text}")
        image = self.txt2img_pipe(text, guidance_scale=self.guidance_scale).images[0]
        
        # Save Text to Image result
        debug_dir = os.path.join("results", "instant_mesh_stages")
        os.makedirs(debug_dir, exist_ok=True)
        filename_prefix = text.replace(" ", "_")
        image.save(os.path.join(debug_dir, f"{filename_prefix}_1_text_to_image.png"))
        
        
        # 2. Preprocess Image
        print("Preprocessing image...")
        processed_image = self.preprocess(image, do_remove_background=True)

        # Save Preprocessed Image
        processed_image.save(os.path.join(debug_dir, f"{filename_prefix}_2_preprocessed.png"))
        
        
        # 3. Generate Multi-View Images
        print("Generating multi-view images...")
        mv_images = self.generate_mvs(processed_image)

        # Save Multi-view Images
        mv_images.save(os.path.join(debug_dir, f"{filename_prefix}_3_multiview.png"))
        
        
        # 4. Reconstruct 3D Mesh
        print("Reconstructing 3D mesh...")
        
        # Prepare input for InstantMesh
        mv_images_np = np.asarray(mv_images, dtype=np.float32) / 255.0
        mv_images_tensor = torch.from_numpy(mv_images_np).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        mv_images_tensor = rearrange(mv_images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)    # (6, 3, 320, 320)
        
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)
        mv_images_tensor = mv_images_tensor.unsqueeze(0).to(self.device)
        
        os.makedirs(output_dir, exist_ok=True)
        mesh_basename = text.replace(' ', '_')
        mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
        
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

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
        results = []
        for text in texts:
            try:
                path = self.convert_text_to_3d(text, output_dir)
                results.append(path)
            except Exception as e:
                print(f"Failed to convert '{text}': {e}")
        return results
