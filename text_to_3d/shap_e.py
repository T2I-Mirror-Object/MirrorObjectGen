from text_to_3d.text_to_3d import TextTo3D
from typing import List

import torch
import os
import numpy as np
import trimesh
from trimesh import transformations

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh



class ShapE(TextTo3D):
    class Model:
        def __init__(self, model, diffusion, xm):
            self.model = model
            self.diffusion = diffusion
            self.xm = xm

    def __init__(
        self, 
        seed: int = 42, 
        guidance: float = 10.0, 
        fp16: bool = True, 
        device: str = "cuda", 
        orientation: List[float] = [0.0, 0.0, 0.0],
        karras_steps: int = 64,
        sigma_min: float = 1e-3,
        sigma_max: float = 160,
        s_churn: float = 0
    ):
        self.seed = seed
        self.guidance = guidance
        self.fp16 = fp16
        self.device = torch.device(device)
        self.orientation = orientation  # [rotation_x, rotation_y, rotation_z] in degrees
        self.karras_steps = karras_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn
        self.model: "ShapE.Model" = None

    def init_model(self):
        torch.manual_seed(self.seed)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # Load Shap-E transmitter & text model + diffusion config
        xm = load_model("transmitter", device=device)            # decoder
        model = load_model("text300M", device=device)            # text encoder
        diffusion = diffusion_from_config(load_config("diffusion"))

        self.model = self.Model(model, diffusion, xm)

    def convert_text_to_3d(self, text: str, output_dir: str) -> str:
        if self.model is None:
            self.init_model()

        latents = sample_latents(
            batch_size=1,
            model=self.model.model,
            diffusion=self.model.diffusion,
            guidance_scale=self.guidance,
            model_kwargs=dict(texts=[text]),
            progress=True,
            clip_denoised=True,
            use_fp16=self.fp16,
            use_karras=True,
            karras_steps=self.karras_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=self.s_churn,
            device=self.device,
        )

        # Decode to a triangle mesh
        tri = decode_latent_mesh(self.model.xm, latents[0]).tri_mesh()

        obj_path = os.path.join(output_dir, f"{text}.obj")
        os.makedirs(output_dir, exist_ok=True)

        # Convert shap-e mesh to trimesh directly
        # Access vertices and faces from the shap-e tri_mesh object
        vertices = tri.verts
        faces = tri.faces
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Apply rotation if orientation is specified
        if any(angle != 0.0 for angle in self.orientation):
            # Convert degrees to radians and create rotation matrix from Euler angles (x, y, z order)
            # orientation: [rotation_x, rotation_y, rotation_z] in degrees
            rotation_x_rad = np.deg2rad(self.orientation[0])
            rotation_y_rad = np.deg2rad(self.orientation[1])
            rotation_z_rad = np.deg2rad(self.orientation[2])
            rotation_matrix = transformations.euler_matrix(
                rotation_x_rad,  # rotation around x-axis
                rotation_y_rad,  # rotation around y-axis
                rotation_z_rad,  # rotation around z-axis
                axes='xyz'
            )
            mesh.apply_transform(rotation_matrix)
        
        # Export the rotated mesh
        mesh.export(obj_path)

        return obj_path
            

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
        if self.model is None:
            self.init_model()

        latents = sample_latents(
            batch_size=len(texts),
            model=self.model.model,
            diffusion=self.model.diffusion,
            guidance_scale=self.guidance,
            model_kwargs=dict(texts=texts),
            progress=True,
            clip_denoised=True,
            use_fp16=self.fp16,
            use_karras=True,
            karras_steps=self.karras_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=self.s_churn,
            device=self.device,
        )

        # Decode to triangle meshes
        tris = [decode_latent_mesh(self.model.xm, latents[i]).tri_mesh() for i in range(len(texts))]

        obj_paths = [os.path.join(output_dir, f"{texts[i]}.obj") for i in range(len(texts))]
        os.makedirs(output_dir, exist_ok=True)

        # Apply rotation to each mesh
        for i in range(len(texts)):
            # Convert shap-e mesh to trimesh directly
            # Access vertices and faces from the shap-e tri_mesh object
            vertices = tris[i].verts
            faces = tris[i].faces
            
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Apply rotation if orientation is specified
            if any(angle != 0.0 for angle in self.orientation):
                # Convert degrees to radians and create rotation matrix from Euler angles (x, y, z order)
                # orientation: [rotation_x, rotation_y, rotation_z] in degrees
                rotation_x_rad = np.deg2rad(self.orientation[0])
                rotation_y_rad = np.deg2rad(self.orientation[1])
                rotation_z_rad = np.deg2rad(self.orientation[2])
                rotation_matrix = transformations.euler_matrix(
                    rotation_x_rad,  # rotation around x-axis
                    rotation_y_rad,  # rotation around y-axis
                    rotation_z_rad,  # rotation around z-axis
                )
                mesh.apply_transform(rotation_matrix)
            
            # Export the rotated mesh
            mesh.export(obj_paths[i])

        return obj_paths