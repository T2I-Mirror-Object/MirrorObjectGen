import argparse
import cv2
import torch
import os
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def get_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Inference Script')
    
    # Input and Output arguments
    parser.add_argument('--input-image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save results')
    
    # Model arguments (optional, but good practice to allow changing)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/depth_anything_v2_vitl.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # 2. Load Model
    # Map encoder types to their feature/channel configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    config = model_configs[args.encoder]
    model = DepthAnythingV2(**config)
    
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model.to(device).eval()

    # 3. Load Image
    if not os.path.exists(args.input_image):
        print(f"Error: Input file {args.input_image} does not exist.")
        return

    raw_img = cv2.imread(args.input_image)
    if raw_img is None:
        print("Error: Failed to load image using cv2.")
        return

    print(f"Processing: {args.input_image}...")

    # 4. Inference
    # The model handles resizing internally, but input should be RGB usually, cv2 is BGR
    # Note: DepthAnything implementation usually handles BGR/RGB, checking source is recommended.
    # Assuming standard implementation handles it, otherwise: cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    depth = model.infer_image(raw_img) # Returns HxW float32 numpy array

    # 5. Post-Processing
    # Normalize depth to 0-255 for visualization
    # We use min-max normalization to stretch the values across the spectrum
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_uint8 = depth_normalized.astype(np.uint8)

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