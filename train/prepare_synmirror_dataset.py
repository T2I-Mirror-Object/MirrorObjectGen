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
        m = (m >= 0.5).astype(np.uint8) * 255
    return m

def _save_image(img_array, path, mode=None):
    """Save numpy array as image."""
    if mode:
        img = Image.fromarray(img_array, mode=mode)
    else:
        img = Image.fromarray(img_array)
    img.save(path)

def get_csv_key_from_path(hdf5_path):
    """
    Robustly extracts the relative path starting from 'abo_v3' for CSV lookup.
    Example: .../SynMirror/abo_v3/0/Item/0.hdf5 -> abo_v3/0/Item/0.hdf5
    """
    parts = list(hdf5_path.parts)
    try:
        # Find where 'abo_v3' starts in the path
        start_index = parts.index("abo_v3")
        # Join everything from abo_v3 onwards
        rel_path = "/".join(parts[start_index:])
        return rel_path
    except ValueError:
        # Fallback if folder structure is unexpected
        return str(hdf5_path)

def main():
    parser = argparse.ArgumentParser(description="Prepare SynMirror Dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory (e.g. SynMirror/abo_v3/0)")
    parser.add_argument("--output_dir", type=str, default="synmirror_dataset", help="Path to output dataset directory")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the abo_split_all.csv file")
    parser.add_argument("--start_idx", type=int, default=0, help="Index to start file naming from (e.g. 0, 1000, etc.)")
    
    args = parser.parse_args()

    # Configuration
    input_dir = Path(args.input_dir)
    csv_path = Path(args.csv_path)
    start_idx = args.start_idx
    
    # Output directories
    output_root = Path(args.output_dir)
    images_dir = output_root / "images"
    depth_dir = output_root / "depth"
    mirror_masks_dir = output_root / "mirror_masks"
    object_masks_dir = output_root / "object_masks"
    metadata_path = output_root / "metadata.jsonl"
    
    # Ensure directories exist
    for d in [images_dir, depth_dir, mirror_masks_dir, object_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load Depth Anything 3 Model
    print("Initializing Depth Anything 3 Estimator...")
    depth_estimator = get_depth_estimator()
    
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Create a dictionary for fast lookup: path -> caption
    # Normalize paths in CSV to ensure consistency (forward slashes)
    path_to_caption = dict(zip(df['path'].apply(lambda x: str(x).replace("\\", "/")), df['caption']))

    print(f"Loaded {len(path_to_caption)} captions.")
    
    # Traverse directory
    print(f"Scanning {input_dir} for .hdf5 files...")
    # Using sorted() for consistent ordering within this batch
    hdf5_files = sorted(list(input_dir.rglob("*.hdf5")))
    
    print(f"Found {len(hdf5_files)} HDF5 files. Processing starting from ID {start_idx+1:05d}...")
    
    # Open metadata file in APPEND mode ('a')
    with open(metadata_path, 'a') as meta_file:
        for idx, hdf5_path in enumerate(tqdm(hdf5_files)):
            try:
                # Calculate current file ID based on loop index + start argument
                current_id = idx + start_idx + 1
                file_id = f"{current_id:05d}"
                
                # 1. Get Caption Key
                # We need the key to look like "abo_v3/X/Object/0.hdf5" regardless of input_dir
                csv_key = get_csv_key_from_path(hdf5_path)
                
                # 2. Extract Data
                data = extract_data_from_hdf5(str(hdf5_path))
                rgb_image = data["image"]
                mirror_mask = _to_numpy_mask(data["mirror_mask"])
                object_mask = _to_numpy_mask(data["object_mask"])
                
                # 3. Get Caption
                caption = path_to_caption.get(csv_key)
                
                if caption is None:
                    # Fallback: Try key without extension
                    csv_key_no_ext = str(Path(csv_key).with_suffix(''))
                    caption = path_to_caption.get(csv_key_no_ext)
                
                if caption is None:
                    caption = "object" # Fallback
                    
                prompt = f"{caption} in front of a mirror, both the object and its perfect mirror reflection are visible"
                
                # 4. Save RGB
                rgb_filename = f"{file_id}.jpg"
                rgb_path = images_dir / rgb_filename
                _save_image(rgb_image, rgb_path)
                
                # 5. Save Masks
                mirror_mask_filename = f"{file_id}.png"
                mirror_mask_path = mirror_masks_dir / mirror_mask_filename
                _save_image(mirror_mask, mirror_mask_path, mode="L")
                
                object_mask_filename = f"{file_id}.png"
                object_mask_path = object_masks_dir / object_mask_filename
                _save_image(object_mask, object_mask_path, mode="L")
                
                # 6. Extract Depth
                prediction = depth_estimator.extract_depth(rgb_image)
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
                
                depth_filename = f"{file_id}.png"
                depth_path = depth_dir / depth_filename
                Image.fromarray(depth_uint16, mode='I;16').save(depth_path)
                
                # 7. Write Metadata
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
                
    print(f"Batch complete. Next start_idx should be: {start_idx + len(hdf5_files)}")

if __name__ == "__main__":
    main()
