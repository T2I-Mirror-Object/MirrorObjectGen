import os
import sys
import torch
import rembg
import numpy as np
from PIL import Image
from typing import List
from torchvision.transforms import v2
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'InstantMesh'))

from text_to_3d.text_to_3d import TextTo3D
from text_to_image.stable_diffusion import StableDiffusionGenerator
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground


class InstantMeshTextTo3D(TextTo3D):
    def __init__(
        self,
        config_path: str,
        output_path: str = 'outputs/',
        sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda",
        diffusion_steps: int = 75,
        scale: float = 1.0,
        view: int = 6,
        export_texmap: bool = False,
        no_rembg: bool = False,
    ):
        """
        Initialize the InstantMesh Text-to-3D pipeline.

        Args:
            config_path: Path to InstantMesh config file
            output_path: Output directory for generated files
            sd_model_id: Stable Diffusion model ID
            device: Device to run on ('cuda' or 'cpu')
            diffusion_steps: Number of denoising steps for InstantMesh
            scale: Scale of generated object
            view: Number of input views (4 or 6)
            export_texmap: Whether to export texture map
            no_rembg: Whether to skip background removal
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_path = output_path
        self.diffusion_steps = diffusion_steps
        self.scale = scale
        self.view = view
        self.export_texmap = export_texmap
        self.no_rembg = no_rembg

        # Load config
        self.config = OmegaConf.load(config_path)
        self.config_name = os.path.basename(config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        self.is_flexicubes = True if self.config_name.startswith('instant-mesh') else False

        # Initialize Stable Diffusion for text-to-image
        print('Initializing Stable Diffusion for text-to-image...')
        self.sd_generator = StableDiffusionGenerator(device=str(self.device), model_id=sd_model_id)

        # Initialize InstantMesh diffusion pipeline for multiview generation
        print('Loading InstantMesh diffusion model...')
        self.im_pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        self.im_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.im_pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # Load custom white-background UNet
        print('Loading custom white-background unet...')
        if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model"
            )
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.im_pipeline.unet.load_state_dict(state_dict, strict=True)
        self.im_pipeline = self.im_pipeline.to(self.device)

        # Load reconstruction model
        print('Loading reconstruction model...')
        self.model = instantiate_from_config(self.model_config)
        if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename=f"{self.config_name.replace('-', '_')}.ckpt",
                repo_type="model"
            )
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

        if self.is_flexicubes:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

        # Initialize rembg session
        self.rembg_session = None if self.no_rembg else rembg.new_session()

        # Setup input cameras
        self.input_cameras = get_zero123plus_input_cameras(
            batch_size=1, radius=4.0 * self.scale
        ).to(self.device)

        print('InstantMesh Text-to-3D pipeline initialized successfully!')

    def _generate_multiview_from_image(self, input_image: Image.Image) -> torch.Tensor:
        """
        Generate multiview images from a single input image.

        Args:
            input_image: Input PIL Image

        Returns:
            Tensor of multiview images (6, 3, 320, 320)
        """
        # Remove background and resize if needed
        if not self.no_rembg:
            input_image = remove_background(input_image, self.rembg_session)
            input_image = resize_foreground(input_image, 0.85)

        # Generate multiview images
        output_image = self.im_pipeline(
            input_image,
            num_inference_steps=self.diffusion_steps,
        ).images[0]

        # Convert to tensor and rearrange
        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)

        return images

    def _reconstruct_mesh(self, images: torch.Tensor, output_name: str, mesh_dir: str) -> str:
        """
        Reconstruct 3D mesh from multiview images.

        Args:
            images: Multiview images tensor (6, 3, 320, 320)
            output_name: Name for output mesh file
            mesh_dir: Directory to save mesh

        Returns:
            Path to saved mesh file
        """
        images = images.unsqueeze(0).to(self.device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        # Select subset of views if needed
        if self.view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
            images = images[:, indices]
            input_cameras = self.input_cameras[:, indices]
        else:
            input_cameras = self.input_cameras

        with torch.no_grad():
            # Get triplane representation
            planes = self.model.forward_planes(images, input_cameras)

            # Extract mesh
            mesh_path = os.path.join(mesh_dir, f'{output_name}.obj')
            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=self.export_texmap,
                **self.infer_config,
            )

            if self.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_path)

        return mesh_path

    def convert_text_to_3d(self, text: str, output_name: str = None) -> str:
        """
        Convert text to 3D mesh.

        Args:
            text: Text description of the object
            output_name: Name for output files (default: uses text as name)

        Returns:
            Path to the generated mesh file
        """
        if output_name is None:
            output_name = text.replace(' ', '_')[:50]  # Use first 50 chars of text

        # Create output directories
        image_dir = os.path.join(self.output_path, self.config_name, 'images')
        mesh_dir = os.path.join(self.output_path, self.config_name, 'meshes')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        print(f'Converting text to 3D: "{text}"')

        # Step 1: Text to Image using Stable Diffusion
        print('Step 1: Generating image from text...')
        image_path = os.path.join(image_dir, f'{output_name}_initial.png')
        input_image = self.sd_generator.generate(text, output_path=image_path)

        # Step 2: Generate multiview images
        print('Step 2: Generating multiview images...')
        multiview_images = self._generate_multiview_from_image(input_image)

        # Save multiview image
        multiview_array = rearrange(
            multiview_images, '(n m) c h w -> c (n h) (m w)', n=3, m=2
        ).permute(1, 2, 0).numpy()
        multiview_pil = Image.fromarray((multiview_array * 255).astype(np.uint8))
        multiview_path = os.path.join(image_dir, f'{output_name}_multiview.png')
        multiview_pil.save(multiview_path)
        print(f'Multiview image saved to {multiview_path}')

        # Step 3: Reconstruct 3D mesh
        print('Step 3: Reconstructing 3D mesh...')
        mesh_path = self._reconstruct_mesh(multiview_images, output_name, mesh_dir)
        print(f'Mesh saved to {mesh_path}')

        return mesh_path

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str = None) -> List[str]:
        """
        Convert multiple texts to 3D objects and save them to directories.

        Args:
            texts: List of text descriptions
            output_dir: Output directory (optional, uses default if not provided)

        Returns:
            List of paths to generated mesh files
        """
        if output_dir:
            self.output_path = output_dir

        mesh_paths = []
        for idx, text in enumerate(texts):
            print(f'\n[{idx+1}/{len(texts)}] Processing: "{text}"')
            output_name = f'text_{idx:03d}_{text.replace(" ", "_")[:30]}'
            try:
                mesh_path = self.convert_text_to_3d(text, output_name=output_name)
                mesh_paths.append(mesh_path)
            except Exception as e:
                print(f'Error processing "{text}": {e}')
                mesh_paths.append(None)

        return mesh_paths
