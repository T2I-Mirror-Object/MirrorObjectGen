import argparse
import cv2
import torch
import os
import numpy as np
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.depth_estimation import get_depth_estimator

def get_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Inference Script (Transformers)')
    
    # Input and Output arguments
    parser.add_argument('--input-image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg', 'v2'],
                        help='Model encoder type or "v2" for default')
    
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # 2. Load Model
    try:
        estimator = get_depth_estimator(model_type=args.encoder, device=device)
    except Exception as e:
        print(f"Failed to load estimator: {e}")
        return

    # 3. Load Image
    if not os.path.exists(args.input_image):
        print(f"Error: Input file {args.input_image} does not exist.")
        return

    print(f"Processing: {args.input_image}...")
    
    # 4. Inference
    # Our util handles loading/conversion
    depth_uint8 = estimator.extract_depth(args.input_image) # HxW uint8

    # 5. Post-Processing
    # Normalize depth to 0-255 for visualization
    # uint8 is already 0-255 normalized
    
    # Generate Colored Heatmap (Inferno is standard for depth)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # 6. Save Results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create output filenames based on input filename
    filename = os.path.basename(args.input_image)
    name, ext = os.path.splitext(filename)

    grayscale_path = os.path.join(args.output_dir, f"{name}_depth_gray.png")
    heatmap_path = os.path.join(args.output_dir, f"{name}_depth_color.png")

    cv2.imwrite(grayscale_path, depth_uint8)
    cv2.imwrite(heatmap_path, depth_colormap)

    print(f"Saved Grayscale: {grayscale_path}")
    print(f"Saved Heatmap:   {heatmap_path}")

if __name__ == '__main__':
    main()