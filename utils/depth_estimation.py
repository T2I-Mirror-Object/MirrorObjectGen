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
            depth_uint16: [H, W] uint16 array
        """
        # Prepare input for pipeline. Pipeline handles str (path) and PIL Image nicely.
        # If numpy, convert to PIL.
        
        pil_image = None
        if isinstance(image, str):
            # Pipeline can handle paths, but let's load it to be consistent with other flows
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR if coming from cv2, or RGB?
            # Standard utils usually imply BGR from cv2, RGB from PIL.
            # Let's assume RGB if 3 channels for safety with PIL conversion, 
            # BUT earlier code handled cv2 BGR.
            # If input is BGR (common in cv2 workflows), we should convert to RGB.
            # Let's try to detect or just assume RGB if passed as numpy to this generic util?
            # Or assume BGR because cv2 is common in this codebase?
            # The previous V2 implementation I wrote explicitly did cv2.imread (BGR) and passed to model.
            # Here pipeline expects PIL (RGB).
            # If the user passes numpy, it's ambiguous.
            # Let's assume RGB for now as `extract_hdf5` returns RGB.
            # If it looks like BGR (e.g. from cv2.imread), user usually converts or we should.
            # `prepare_synmirror_dataset.py` calls `extract_data_from_hdf5` which returns RGB (PIL -> np).
            # So RGB is the safe bet for `prepare_synmirror_dataset.py`.
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
        
        # The pipeline output 'depth' is usually already a visualized depth map (uint8?) or raw values?
        # For "depth-estimation", transformers usually returns the depth map as a PIL Image.
        # If it's a "depth-anything" model, the output might be raw depth if using model directly,
        # but pipeline often normalizes for visualization.
        # Let's inspect the type. If it's floating point, great. If uint8, we are limited.
        # Usually HF depth pipeline returns PIL Image in mode "L" (8-bit) or "I" (32-bit int) or "F" (32-bit float)?
        # Documentation says "a PIL Image".
        # Let's assume we get something reasonable. We need to cast to uint16 0-65535.
        
        depth_map = depth_map.astype(np.float32)
        
        # Normalize to 0-65535
        d_min = depth_map.min()
        d_max = depth_map.max()
        
        if d_max - d_min > 1e-8:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_map)
            
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
        return depth_uint16

def get_depth_estimator(model_type="v2", device=None):
    """Factory function to get a Depth Estimator instance."""
    if model_type == "v2" or model_type in ['vitl', 'vitb', 'vits', 'vitg']:
        return DepthAnythingV2Estimator(model_type=model_type, device=device)
    else:
        return DepthAnything3Estimator(device=device)
