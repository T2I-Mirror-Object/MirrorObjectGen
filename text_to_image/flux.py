import torch
from diffusers import FluxPipeline
from PIL import Image
import os

class FluxGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "black-forest-labs/FLUX.1-dev"):
        # Note: FluxPipeline might handle device placement slightly differently with offloading, 
        # but we'll stick to a common pattern where possible or defer to the pipeline's defaults.
        # The script used enable_model_cpu_offload(), which is typical for large models like Flux.
        
        print(f"Loading FLUX.1 model: {model_id}...")
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        # Enable CUDA optimizations for memory efficiency as seen in the script
        self.pipe.enable_model_cpu_offload()
        
    def generate(self, 
                 prompt: str, 
                 output_path: str = None,
                 height: int = 1024,
                 width: int = 1024,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 3.5,
                 seed: int = 0
                 ) -> Image.Image:
        """
        Generate an image from a text prompt using FLUX.1.
        
        Args:
            prompt: Text description of the image.
            output_path: Optional path to save the generated image.
            height: Height of the generated image.
            width: Width of the generated image.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Guidance scale (CFG).
            seed: Random seed for reproducibility.
            
        Returns:
            The generated PIL Image.
        """
        print(f"Generating image for: '{prompt}'")
        
        generator = torch.Generator("cpu").manual_seed(seed)
        
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            print(f"Image saved to {output_path}")
            
        return image
