import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch

from utils.extract_hdf5 import extract_data_from_hdf5

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

def main():
    # Configuration
    data_root = Path("SynMirror")
    abo_v3_root = data_root / "abo_v3"
    csv_path = data_root / "abo_split_all.csv"
    
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Create a dictionary for fast lookup: path -> caption
    # The 'path' in CSV is 'abo_v3/B/B07JY4H14B/0'
    # We will normalize it to match our file traversal if needed.
    path_to_caption = dict(zip(df['path'], df['caption']))

    print(len(path_to_caption))
    print(path_to_caption["abo_v3/1/B075X3S2Z1/0.hdf5"])
    
    # Traverse directory
    print(f"Scanning {abo_v3_root} for .hdf5 files...")
    hdf5_files = list(abo_v3_root.rglob("*.hdf5"))
    
    print(f"Found {len(hdf5_files)} HDF5 files. Processing...")
    
    for hdf5_path in tqdm(hdf5_files):
        # hdf5_path is absolute or relative to cwd. 
        # We need to construct the key for path_to_caption.
        # The key in CSV is relative to data/SynMirror, e.g., abo_v3/X/ID/0
        # hdf5_path relative to data_root would be abo_v3/X/ID/0.hdf5
        
        rel_path = hdf5_path.relative_to(data_root)

        csv_key = str(rel_path)
        
        # 1. Extract Data
        data = extract_data_from_hdf5(str(hdf5_path))
        
        # 2. Get Caption
        caption = path_to_caption.get(csv_key)

if __name__ == "__main__":
    main()
