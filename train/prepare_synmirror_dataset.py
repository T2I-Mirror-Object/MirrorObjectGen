import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import json
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract_hdf5 import extract_data_from_hdf5
from utils.depth_estimation import get_depth_estimator

def _to_numpy_mask(m):
    """Convert mask to uint8 [0,255] HxW array."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    else:
        m = np.array(m)
    m = np.squeeze(m)

    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        # If it's not uint8, assume it's some other numeric type (e.g. float 0-1 or similar)
        # For masks, we typically want 0 or 255.
        m = (m >= 0.5).astype(np.uint8) * 255
    return m

def _save_image(img_array, path, mode=None):
    """Save numpy array as image."""
    if mode:
        img = Image.fromarray(img_array, mode=mode)
    else:
        img = Image.fromarray(img_array)
    img.save(path)

def main():
    parser = argparse.ArgumentParser(description="Prepare SynMirror Dataset")
    parser.add_argument("--input_dir", type=str, default="SynMirror", help="Path to input SynMirror directory")
    parser.add_argument("--output_dir", type=str, default="synmirror_dataset", help="Path to output dataset directory")
    args = parser.parse_args()

    # Configuration
    data_root = Path(args.input_dir)
    abo_v3_root = data_root / "abo_v3"
    csv_path = data_root / "abo_split_all.csv"
    
    # Output directories
    output_root = Path(args.output_dir)
    images_dir = output_root / "images"
    depth_dir = output_root / "depth"
    mirror_masks_dir = output_root / "mirror_masks"
    object_masks_dir = output_root / "object_masks"
    metadata_path = output_root / "metadata.jsonl"
    
    for d in [images_dir, depth_dir, mirror_masks_dir, object_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load Depth Anything 3 Model
    print("Initializing Depth Anything 3 Estimator...")
    depth_estimator = get_depth_estimator()
    
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Create a dictionary for fast lookup: path -> caption
    path_to_caption = dict(zip(df['path'], df['caption']))

    print(f"Loaded {len(path_to_caption)} captions.")
    
    # Traverse directory
    print(f"Scanning {abo_v3_root} for .hdf5 files...")
    # Using sorted() for consistent ordering
    hdf5_files = sorted(list(abo_v3_root.rglob("*.hdf5")))
    
    print(f"Found {len(hdf5_files)} HDF5 files. Processing...")
    
    # Open metadata file
    with open(metadata_path, 'w') as meta_file:
        for idx, hdf5_path in enumerate(tqdm(hdf5_files)):
            try:
                # Construct file ID: 00001, 00002, etc. (1-indexed)
                file_id = f"{idx+1:05d}"
                
                # Rel path for CSV lookup
                rel_path = hdf5_path.relative_to(data_root)
                csv_key = str(rel_path).replace("\\", "/") # Ensure forward slashes for dictionary lookup
                
                # 1. Extract Data
                data = extract_data_from_hdf5(str(hdf5_path))
                rgb_image = data["image"]
                mirror_mask = _to_numpy_mask(data["mirror_mask"])
                object_mask = _to_numpy_mask(data["object_mask"])
                
                # 2. Get Caption
                # Helper: Normalize key if exact match fails (remove 'SynMirror/' if present, or adjust extension)
                # The csv_key from user context seemed to be 'abo_v3/...' which matches rel_path if data_root is parent of abo_v3
                caption = path_to_caption.get(csv_key)
                
                if caption is None:
                    # Fallback or skip? Let's use a generic caption if missing, or specific logic.
                    # Try removing extension .hdf5?
                    csv_key_no_ext = str(rel_path.with_suffix('')).replace("\\", "/")
                    caption = path_to_caption.get(csv_key_no_ext)
                
                if caption is None:
                    caption = "object" # Fallback
                    
                prompt = f"{caption} in front of a mirror, both the object and its perfect mirror reflection are visible"
                
                # 3. Save RGB
                rgb_filename = f"{file_id}.jpg"
                rgb_path = images_dir / rgb_filename
                _save_image(rgb_image, rgb_path)
                
                # 4. Save Masks
                mirror_mask_filename = f"{file_id}.png"
                mirror_mask_path = mirror_masks_dir / mirror_mask_filename
                _save_image(mirror_mask, mirror_mask_path, mode="L")
                
                object_mask_filename = f"{file_id}.png"
                object_mask_path = object_masks_dir / object_mask_filename
                _save_image(object_mask, object_mask_path, mode="L")
                
                # 5. Extract Depth
                # Model expects list of images or numpy array
                # rgb_image is HxWxC uint8
                prediction = depth_estimator.extract_depth(rgb_image)
                depth_map = prediction.depth  # [N, H, W] -> [1, H, W]
                # Squeeze to [H, W]
                depth_map = depth_map[0]
                
                # Normalize depth for saving (16-bit PNG)
                # Depth Anything returns relative depth (metric if configured, but usually relative inverse depth)
                # We normalize min-max to 0-65535 for visibility/storage
                d_min = depth_map.min()
                d_max = depth_map.max()
                
                if d_max - d_min > 1e-8:
                    depth_normalized = (depth_map - d_min) / (d_max - d_min)
                    # Invert depth: White (1.0) is Near, Black (0.0) is Far
                    depth_normalized = 1.0 - depth_normalized
                else:
                    depth_normalized = np.zeros_like(depth_map)
                    
                depth_uint16 = (depth_normalized * 65535).astype(np.uint16)
                
                depth_filename = f"{file_id}.png"
                depth_path = depth_dir / depth_filename
                # PIL mode 'I;16' handles 16-bit grayscale
                Image.fromarray(depth_uint16, mode='I;16').save(depth_path)
                
                # 6. Write Metadata
                metadata_entry = {
                    "image": f"images/{rgb_filename}",
                    "depth": f"depth/{depth_filename}",
                    "mirror_mask": f"mirror_masks/{mirror_mask_filename}",
                    "object_mask": f"object_masks/{object_mask_filename}",
                    "text": prompt
                }
                
                meta_file.write(json.dumps(metadata_entry) + "\n")
                
            except Exception as e:
                print(f"Error processing {hdf5_path}: {e}")
                continue
                
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
