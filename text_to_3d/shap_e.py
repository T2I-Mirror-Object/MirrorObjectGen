from text_to_3d.text_to_3d import TextTo3D
from typing import List

import torch
import os, gc

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

    def __init__(self, seed: int = 42, guidance: float = 10.0, fp16: bool = True, device: str = "cuda"):
        self.seed = seed
        self.guidance = guidance
        self.fp16 = fp16
        self.device = torch.device(device)
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

        prompt = text + " facing ahead"

        latents = sample_latents(
            batch_size=1,
            model=self.model.model,
            diffusion=self.model.diffusion,
            guidance_scale=self.guidance,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=self.fp16,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
            device=self.device,
        )

        # Decode to a triangle mesh and write to disk
        tri = decode_latent_mesh(self.model.xm, latents[0]).tri_mesh()

        obj_path = os.path.join(output_dir, f"{text}.obj")
        os.makedirs(output_dir, exist_ok=True)

        with open(obj_path, "w") as f:
            tri.write_obj(f)

        return obj_path
            

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
        if self.model is None:
            self.init_model()
        prompt = [text + " facing ahead" for text in texts]

        latents = sample_latents(
            batch_size=len(texts),
            model=self.model.model,
            diffusion=self.model.diffusion,
            guidance_scale=self.guidance,
            model_kwargs=dict(texts=prompt),
            progress=True,
            clip_denoised=True,
            use_fp16=self.fp16,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
            device=self.device,
        )

        # Decode to a triangle mesh and write to disk
        tris = [decode_latent_mesh(self.model.xm, latents[i]).tri_mesh() for i in range(len(texts))]

        obj_paths = [os.path.join(output_dir, f"{texts[i]}.obj") for i in range(len(texts))]
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(texts)):
            with open(obj_paths[i], "w") as f:
                tris[i].write_obj(f)

        return obj_paths