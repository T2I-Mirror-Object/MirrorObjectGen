import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class StableDiffusionGenerator:
    def __init__(self, device: str = "cuda", model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading Stable Diffusion model: {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.pipe.to(self.device)

    def generate(self, prompt: str, output_path: str = None) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image.
            output_path: Optional path to save the generated image.
            
        Returns:
            The generated PIL Image.
        """
        print(f"Generating image for: '{prompt}'")
        image = self.pipe(prompt).images[0]
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            print(f"Image saved to {output_path}")
            
        return image
