import h5py
import numpy as np
from PIL import Image
from pathlib import Path

def extract_data_from_hdf5(hdf5_path: str, output_path: str):
    """returns the data present in the hdf5_path file"""

    hdf5_data = h5py.File(hdf5_path, "r")

    colors = np.array(hdf5_data["colors"], dtype=np.uint8),

    img = Image.fromarray(colors[0])
    img.save(output_path)

def extract_first_hdf5_from_all_directories(base_path: str, output_base_path: str):
    """
    Find all directories in hf-objaverse-v4 and extract the first hdf5 file (0.hdf5) 
    from each subdirectory.
    
    Args:
        base_path: Path to hf-objaverse-v4 directory
        output_base_path: Base path where extracted images will be saved
    """
    base_path = Path(base_path)
    output_base_path = Path(output_base_path)
    
    # Create output directory if it doesn't exist
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Get all directories in hf-objaverse-v4 (e.g., 000-030, 000-032, etc.)
    directories = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(directories)} directories in {base_path}")
    
    total_extracted = 0
    total_errors = 0
    
    for directory in directories:
        print(f"\nProcessing directory: {directory.name}")
        
        # Get all subdirectories in this directory
        subdirectories = [sd for sd in directory.iterdir() if sd.is_dir()]
        
        print(f"  Found {len(subdirectories)} subdirectories")
        
        # Create output directory for this main directory
        output_dir = output_base_path / directory.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in subdirectories:
            # Path to the first hdf5 file (0.hdf5)
            hdf5_file = subdir / "0.hdf5"
            
            if hdf5_file.exists():
                try:
                    # Create output path: output_base/000-030/subdir_name.png
                    output_file = output_dir / f"{subdir.name}.png"
                    
                    # Extract and save
                    extract_data_from_hdf5(str(hdf5_file), str(output_file))
                    total_extracted += 1
                    
                    if total_extracted % 100 == 0:
                        print(f"  Extracted {total_extracted} files so far...")
                        
                except Exception as e:
                    print(f"  Error processing {hdf5_file}: {e}")
                    total_errors += 1
            else:
                print(f"  Warning: {hdf5_file} not found")
                total_errors += 1
    
    print("\n\nExtraction complete!")
    print(f"Total extracted: {total_extracted}")
    print(f"Total errors: {total_errors}")

if __name__ == "__main__":
    # Extract from all directories
    base_path = "/Users/xuanbachmai/CS/DeepLearning/hf-objaverse-v4"
    output_path = "dataset"
    
    extract_first_hdf5_from_all_directories(base_path, output_path)