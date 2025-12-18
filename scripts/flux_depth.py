import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate images using FLUX.1-Depth-dev model with depth map conditioning')
parser.add_argument(
    '--prompt',
    type=str,
    required=True,
    help='Text prompt for image generation'
)
parser.add_argument(
    '--model-id',
    type=str,
    default="black-forest-labs/FLUX.1-Depth-dev",
    help='HuggingFace model ID'
)
parser.add_argument(
    '--depth-map',
    type=str,
    default="results/depth/scene_depth.png",
    help='Path to depth map image'
)
parser.add_argument(
    '--lora-path',
    type=str,
    default=None,
    help='Path to LoRA weights'
)
parser.add_argument(
    '--output',
    type=str,
    default="results/flux_depth/output.png",
    help='Output image path'
)
parser.add_argument(
    '--height',
    type=int,
    default=512,
    help='Output image height'
)
parser.add_argument(
    '--width',
    type=int,
    default=512,
    help='Output image width'
)
parser.add_argument(
    '--num-inference-steps',
    type=int,
    default=28,
    help='Number of inference steps'
)
parser.add_argument(
    '--guidance-scale',
    type=float,
    default=3.5,
    help='Guidance scale for generation'
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Random seed for reproducibility'
)
parser.add_argument(
    '--device',
    type=str,
    default="cuda",
    help='Device to use for inference'
)

args = parser.parse_args()

# Create output directory
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FLUX.1-Depth-dev Image Generation")
print("=" * 60)
print(f"Propmp: {args.prompt}")
print(f"Model ID: {args.model_id}")
print(f"Depth map: {args.depth_map}")
print(f"Output: {args.output}")
print(f"Image size: {args.width}x{args.height}")
print(f"Inference steps: {args.num_inference_steps}")
print(f"Guidance scale: {args.guidance_scale}")
print(f"Seed: {args.seed}")
print(f"Device: {args.device}")
print()

# Load the pipeline
print("Loading FLUX.1-Depth-dev model...")

try:
    pipe = FluxControlPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16
    ).to(args.device)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nNote: You may need to accept the model license at:")
    print("https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev")
    print("and login with: huggingface-cli login")
    exit(1)

# Load depth map
print(f"\nLoading depth map from: {args.depth_map}")
try:
    # Load the depth map image
    depth_image = Image.open(args.depth_map)
    
    # Convert grayscale depth map to RGB if needed
    # FLUX.1-Depth-dev expects RGB depth maps
    if depth_image.mode != 'RGB':
        depth_image = depth_image.convert('RGB')
    
    print(f"✓ Depth map loaded: {depth_image.size} ({depth_image.mode})")
except Exception as e:
    print(f"✗ Error loading depth map: {e}")
    exit(1)

# Generate image
print(f"\nGenerating image...")

try:
    image = pipe(
        prompt=args.prompt,
        control_image=depth_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
    ).images[0]
    
    print("✓ Image generated successfully")
except Exception as e:
    print(f"✗ Error during generation: {e}")
    exit(1)

# Save the output
print(f"\nSaving image to: {args.output}")
try:
    image.save(args.output)
    print("✓ Image saved successfully")
except Exception as e:
    print(f"✗ Error saving image: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Generation completed successfully!")
print("=" * 60)
print(f"\nGenerated image saved to: {args.output}")
