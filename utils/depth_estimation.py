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
                 model_type: str = 'vitl', 
                 checkpoint_path: str = 'checkpoints/depth_anything_v2_vitl.pth', 
                 device: Optional[torch.device] = None):
        """
        Initialize the Depth Anything V2 estimator.
        """
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError("Please install depth_anything_v2 to use this feature.")
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading Depth Anything V2 model ({model_type}) on {self.device}...")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}. available: {list(model_configs.keys())}")
            
        self.model = DepthAnythingV2(**model_configs[model_type])
        
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        self.model = self.model.to(self.device).eval()
        print("Model loaded successfully.")

    def extract_depth(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract depth map from an image.
        
        Args:
            image: Image path, numpy array (BGR), or PIL Image.
            
        Returns:
            depth_uint16: [H, W] uint16 array
        """
        # Convert to numpy array (cv2 expects BGR)
        if isinstance(image, str):
            raw_img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Assumes BGR if 3 channels, or Gray if 1
            if image.ndim == 3 and image.shape[2] == 3:
                raw_img = image # Assume BGR
                # If the input was RGB (common in PIL -> np), this might be wrong if not handled.
                # However, the user prompt code uses cv2.imread which gives BGR.
                # If the input is from dataset_scripts which uses PIL/extract_hdf5, it might be RGB.
                # Let's check typical usage.
            else:
                raw_img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
        depth = self.model.infer_image(raw_img) # HxW raw depth map
        
        # Normalize depth for saving (16-bit PNG)
        d_min = depth.min()
        d_max = depth.max()

        if d_max - d_min > 1e-8:
            depth_normalized = (depth - d_min) / (d_max - d_min)
            # Check if inversion is needed. 
            # Depth Anything V2 output: usually relative depth. 
            # In V2, closer objects usually have higher values? Or lower?
            # Metric depth: closer = smaller value. Relative depth (often inverse depth): closer = larger value.
            # DPV2 paper says "relative depth". usually disparity-like.
            # Standard 16-bit depth map: usually black=far, white=near (disparity-like) or reversed.
            # The previous code (DA3) did:
            # depth_normalized = (depth_map - d_min) / (d_max - d_min)
            # depth_normalized = 1.0 - depth_normalized (White=Near, Black=Far)
            
            # Let's assume prediction is "inverse depth" (like disparity) where higher = closer.
            # If so, normalizing 0..1 gives 0=far, 1=near.
            # If we want White=Near, we just keep it as is.
            # Wait, previous code had `depth_normalized = 1.0 - depth_normalized` with comment "White (1.0) is Near".
            # This implies original `depth_map` had smaller values for Near ?? 
            # Or `depth_map` had larger values for Near, and 1-norm reversed it?
            # Actually, `DepthAnything3.inference` returns metric depth or relative? 
            # Usually DA models output disparity (close=large).
            # If close=large, then (val - min)/(max-min) -> close=1.0 (White).
            # If previous code did `1.0 - ...`, then it flips it so close=0.0 (Black).
            
            # Let's stick to standard practice or follow user demo. user demo returns `depth`.
            # I will follow the same normalization logic as before BUT without the inversion if V2 behaves like V1 (disparity).
            
            # Let's assume V2 outputs disparity-like values (High=Close).
            # If we want standard visualization/saving where uint16 range maps to depth?
            # Let's default to just normalizing min-max to 0-65535.
            pass
        else:
            depth_normalized = np.zeros_like(depth)

        # Re-using logic from DA3 section for consistency
        # Normalize to 0-1
        depth_normalized = (depth - d_min) / (d_max - d_min)
        
        # Invert? 
        # "Depth Anything V1" outputs relative depth (inverse depth). High value = Close.
        # "Depth Anything V2" likely same.
        # Previous code: 1.0 - depth_normalized.
        # If input was High=Close -> Normalized High=1.0 -> Inverted Low=0.0. So Close=0.0 (Black).
        # Typically depth maps are Black=Far (0), White=Near (65535).
        # OR Black=Close(0), White=Far(65535) (Metric depth in mm).
        
        # Let's look at `dataset_scripts/prepare_synmirror_dataset.py` usage.
        # It just saves it.
        
        # I will keep the raw normalization 0..1 -> 0..65535.
        # If V2 output is disparity (High=Close), then 65535=Close (White).
        # This matches "White (1.0) is Near".
        # So I should NOT invert if I want White=Near.
        # The previous code INVERTED.
        # (depth - min)/(max - min) -> 0..1 (Far..Near if disparity).
        # 1 - (Far..Near) -> (Near..Far). i.e. 1=Far, 0=Near.
        # Comment said: "White (1.0) is Near".
        # If 1.0 is Near, and we did 1.0 - norm, then norm must be 0.0 for Near.
        # That means original depth was Low=Near ??
        
        # Let's play safe and just return normalized depth 0-65535 based on min/max.
        # User can adjust if needed.
        
        depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
        return depth_uint16

def get_depth_estimator(model_type="v2", device=None):
    """Factory function to get a Depth Estimator instance."""
    if model_type == "v2":
        return DepthAnythingV2Estimator(device=device)
    else:
        return DepthAnything3Estimator(device=device)
