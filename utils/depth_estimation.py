import torch
import os
from PIL import Image
import numpy as np
from typing import List, Union, Optional

import cv2

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
        d_min = depth_map.min()
        d_max = depth_map.max()
        
        if d_max - d_min > 1e-8:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
            # Invert depth: White (1.0) is Near, Black (0.0) is Far
            depth_normalized = 1.0 - depth_normalized
        else:
            depth_normalized = np.zeros_like(depth_map)
            
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
            
        return depth_uint16

class DepthAnythingV2Estimator:
    def __init__(self, 
                 model_type: str = 'depth-anything/Depth-Anything-V2-Large-hf', 
                 device: Optional[torch.device] = None):
        """
        Initialize the Depth Anything V2 estimator using Transformers pipeline.
        Args:
            model_type (str): Hugging Face model hub path.
                              Default: "depth-anything/Depth-Anything-V2-Large-hf"
                              Legacy 'vitl' etc mapped to HF path if possible, or used as is.
            device (torch.device, optional): Device to run the model on.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Please install transformers to use this feature: pip install transformers")
            
        # Map legacy short names to HF model IDs if needed, or use default
        self.model_mapping = {
            'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
            'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
            'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
            'vitg': 'depth-anything/Depth-Anything-V2-Giant-hf' # If available/supported
        }
        
        hf_model_id = self.model_mapping.get(model_type, model_type)
        
        device_id = 0 if (device is not None and device.type == 'cuda') or (device is None and torch.cuda.is_available()) else -1
        
        print(f"Loading Depth Anything V2 pipeline: {hf_model_id} on device {device_id}...")
        self.pipe = pipeline(task="depth-estimation", model=hf_model_id, device=device_id)
        print("Pipeline loaded successfully.")

    def extract_depth(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract depth map from an image.
        
        Args:
            image: Image path, numpy array (BGR/RGB), or PIL Image.
            
        Returns:
            depth_uint8: [H, W] uint8 array
        """
        # Prepare input for pipeline. Pipeline handles str (path) and PIL Image nicely.
        # If numpy, convert to PIL.
        
        pil_image = None
        if isinstance(image, str):
            # Pipeline can handle paths, but let's load it to be consistent with other flows
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume RGB if coming from dataset script or PIL compatible source
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
        # Inference
        # pipe(image)["depth"] returns a PIL Image object
        result = self.pipe(pil_image)
        depth_pil = result["depth"]
        
        # Convert to numpy
        depth_map = np.array(depth_pil)
        
        depth_map = depth_map.astype(np.float32)
        
        # Normalize to 0-255
        d_min = depth_map.min()
        d_max = depth_map.max()
        
        if d_max - d_min > 1e-8:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_map)
            
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return depth_uint8

def get_depth_estimator(model_type="v2", device=None):
    """Factory function to get a Depth Estimator instance."""
    if model_type == "v2" or model_type in ['vitl', 'vitb', 'vits', 'vitg']:
        return DepthAnythingV2Estimator(model_type=model_type, device=device)
    else:
        return DepthAnything3Estimator(device=device)
