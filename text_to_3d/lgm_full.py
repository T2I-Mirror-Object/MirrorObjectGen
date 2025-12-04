from text_to_3d.text_to_3d import TextTo3D
from typing import List
import os
import torch
from diffusers import DiffusionPipeline

class LGMFull(TextTo3D):
    def __init__(self, device: str = "cuda", guidance_scale: float = 7.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.guidance_scale = guidance_scale
        self.lgm_pipe = None

    def init_model(self):
        if self.lgm_pipe is None:
            print("Loading LGM-full model...")
            self.lgm_pipe = DiffusionPipeline.from_pretrained(
                "dylanebert/LGM-full",
                custom_pipeline="dylanebert/LGM-full",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
            self.lgm_pipe.to(self.device)

    def convert_text_to_3d(self, text: str, output_dir: str) -> str:
        self.init_model()
        
        print(f"Generating 3D model for: {text}")
        
        # LGM-full pipeline can take text directly
        # Usage based on user snippet: pipeline(input_prompt, None)
        # It seems the second argument is for image, passing None implies text-to-3d
        result = self.lgm_pipe(text, None)
        
        os.makedirs(output_dir, exist_ok=True)
        ply_path = os.path.join(output_dir, f"{text.replace(' ', '_')}.ply")
        
        # Save using the pipeline's method
        self.lgm_pipe.save_ply(result, ply_path)

        return ply_path

    def convert_multiple_texts_to_3d(self, texts: List[str], output_dir: str) -> List[str]:
        results = []
        for text in texts:
            try:
                path = self.convert_text_to_3d(text, output_dir)
                results.append(path)
            except Exception as e:
                print(f"Failed to convert '{text}': {e}")
        return results
