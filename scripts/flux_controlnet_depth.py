import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate images using FLUX.1-dev with ControlNet-Depth')
parser.add_argument(
    '--prompt',
    type=str,
    required=True,
    help='Text prompt for image generation'
)
parser.add_argument(
    '--control-image',
    type=str,
    default="results/depth/scene_depth.png",
    help='Path to control image (depth map) (default: results/depth/scene_depth.png)'
)
parser.add_argument(
    '--output',
    type=str,
    default="results/flux_controlnet/output.png",
    help='Output image path (default: results/flux_controlnet/output.png)'
)
parser.add_argument(
    '--height',
    type=int,
    default=1024,
    help='Output image height (default: 1024)'
)
parser.add_argument(
    '--width',
    type=int,
    default=1024,
    help='Output image width (default: 1024)'
)
parser.add_argument(
    '--num-inference-steps',
    type=int,
    default=24,
    help='Number of inference steps (default: 24)'
)
parser.add_argument(
    '--guidance-scale',
    type=float,
    default=3.5,
    help='Guidance scale for generation (default: 3.5)'
)
parser.add_argument(
    '--controlnet-conditioning-scale',
    type=float,
    default=0.5,
    help='ControlNet conditioning scale (default: 0.5)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility (default: 42)'
)
parser.add_argument(
    '--device',
    type=str,
    default="cuda",
    help='Device to use for inference (default: cuda)'
)

args = parser.parse_args()

# Create output directory
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FLUX.1-ControlNet-Depth Image Generation")
print("=" * 60)
print(f"\nPrompt: {args.prompt}")
print(f"Control Image: {args.control_image}")
print(f"Output: {args.output}")
print(f"Image size: {args.width}x{args.height}")
print(f"Inference steps: {args.num_inference_steps}")
print(f"Guidance scale: {args.guidance_scale}")
print(f"ControlNet Scale: {args.controlnet_conditioning_scale}")
print(f"Seed: {args.seed}")
print(f"Device: {args.device}")
print()

# Load the pipeline
print("Loading FLUX.1 ControlNet models...")
print("(This may take a while on first run as the models are downloaded)")

try:
    controlnet = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
        torch_dtype=torch.bfloat16
    )
    
    pipe = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to(args.device)
    
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("\nNote: You may need to accept the model license at:")
    print("https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("and login with: huggingface-cli login")
    exit(1)

# Load control image
print(f"\nLoading control image from: {args.control_image}")
try:
    # load_image handles both local paths and URLs
    control_image = load_image(args.control_image)
    
    print(f"✓ Control image loaded: {control_image.size}")
except Exception as e:
    print(f"✗ Error loading control image: {e}")
    exit(1)

# Generate image
print(f"\nGenerating image...")
print(f"This may take a few minutes depending on your hardware...")

try:
    image = pipe(
        prompt=args.prompt,
        control_image=control_image,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        width=args.width,
        height=args.height,
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
