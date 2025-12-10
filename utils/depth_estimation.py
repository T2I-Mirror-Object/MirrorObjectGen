import torch
import os
from PIL import Image
import numpy as np
from typing import List, Union, Optional

class DepthAnything3Estimator:
    def __init__(self, model_name: str = "depth-anything/da3mono-large", device: Optional[torch.device] = None):
        """
        Initialize the Depth Anything 3 estimator.
        
        Args:
            model_name (str): Hugging Face model hub path.
            device (torch.device, optional): Device to run the model on. Defaults to CUDA if available.
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise ImportError("Please install depth_anything_3 to use this feature. "
                              "pip install git+https://github.com/LiheYoung/Depth-Anything")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading Depth Anything 3 model: {model_name} on {self.device}...")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def extract_depth(self, 
                      images: Union[str, List[str], np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]], 
                      export_dir: Optional[str] = None, 
                      export_format: str = "npz"):
        """
        Run inference on images to extract depth maps.
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays, or a single instance of these.
            export_dir (str, optional): Directory to save results.
            export_format (str): Format to export results ('glb', 'npz', 'ply', 'mini_npz', 'gs_ply', 'gs_video').
        
        Returns:
            depth_uint16: [H, W] uint16 array
        """
        if not isinstance(images, list):
            images = [images]
            
        # The inference method handles various input types
        with torch.no_grad():
            prediction = self.model.inference(
                images,
                export_dir=export_dir,
                export_format=export_format
            )
        
        depth_map = prediction.depth 
        depth_map = depth_map[0] # Squeeze
        
        # Normalize depth for saving (16-bit PNG)
        d_min = np.percentile(depth_map, 2)
        d_max = np.percentile(depth_map, 98)

        depth_map = np.clip(depth_map, d_min, d_max)
        
        if d_max - d_min > 1e-8:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
            # Invert depth: White (1.0) is Near, Black (0.0) is Far
            depth_normalized = 1.0 - depth_normalized
        else:
            depth_normalized = np.zeros_like(depth_map)
            
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
            
        return depth_uint16

def get_depth_estimator(model_name="depth-anything/da3mono-large", device=None):
    """Factory function to get a DepthAnything3Estimator instance."""
    return DepthAnything3Estimator(model_name=model_name, device=device)
