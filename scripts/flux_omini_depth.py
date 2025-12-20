import os
import torch
import argparse
import sys
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline

# Add the parent directory to sys.path to allow importing from utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.flux_omini import Condition, generate, seed_everything

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate images using FLUX.1-dev with OminiControl Depth LoRA')
parser.add_argument(
    '--prompt',
    type=str,
    required=True,
    help='Text prompt for image generation'
)
parser.add_argument(
    '--model-id',
    type=str,
    default="black-forest-labs/FLUX.1-dev",
    help='HuggingFace model ID'
)
parser.add_argument(
    '--depth-map',
    type=str,
    required=True,
    help='Path to input depth map image'
)
parser.add_argument(
    '--lora-repo',
    type=str,
    default="Yuanshi/OminiControl",
    help='LoRA repository ID'
)
parser.add_argument(
    '--lora-weight-name',
    type=str,
    default="experimental/depth.safetensors",
    help='LoRA weight filename'
)
parser.add_argument(
    '--adapter-name',
    type=str,
    default="depth",
    help='Adapter name'
)
parser.add_argument(
    '--output',
    type=str,
    default="results/flux_omini_depth/output.png",
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
print("Flux OminiControl Depth Generation")
print("=" * 60)
print(f"Prompt: {args.prompt}")
print(f"Model ID: {args.model_id}")
print(f"Depth map: {args.depth_map}")
print(f"Output: {args.output}")
print(f"Image size: {args.width}x{args.height}")
print(f"LoRA: {args.lora_repo} / {args.lora_weight_name}")
print(f"Inference steps: {args.num_inference_steps}")
print(f"Guidance scale: {args.guidance_scale}")
print(f"Seed: {args.seed}")
print(f"Device: {args.device}")
print()

# Load the pipeline
print(f"Loading model with id: {args.model_id}...")

try:
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16
    ).to(args.device)
    
    # Unload existing LoRA weights if any (sanity check, though new pipeline shouldn't have any)
    pipe.unload_lora_weights()

    print(f"Loading LoRA weights from {args.lora_repo}...")
    pipe.load_lora_weights(
        args.lora_repo,
        weight_name=args.lora_weight_name,
        adapter_name=args.adapter_name,
    )
    pipe.set_adapters([args.adapter_name])
    
    print("✓ Model and LoRA loaded successfully")
except Exception as e:
    print(f"✗ Error loading model or LoRA: {e}")
    exit(1)

# Load depth map
print(f"\nLoading depth map from: {args.depth_map}")
try:
    # Load the depth map image
    depth_image = Image.open(args.depth_map).convert("RGB")
    
    # Resize to target dimensions
    if depth_image.size != (args.width, args.height):
        print(f"Resizing depth map from {depth_image.size} to ({args.width}, {args.height})")
        depth_image = depth_image.resize((args.width, args.height), Image.Resampling.LANCZOS)
    
    print(f"✓ Depth map loaded and processed: {depth_image.size}")
except Exception as e:
    print(f"✗ Error loading depth map: {e}")
    exit(1)

# Create Condition
condition = Condition(depth_image, args.adapter_name)

# Set seed
seed_everything(args.seed)

# Generate image
print(f"\nGenerating image...")

try:
    result_img = generate(
        pipe,
        prompt=args.prompt,
        conditions=[condition],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    ).images[0]
    
    print("✓ Image generated successfully")
except Exception as e:
    print(f"✗ Error during generation: {e}")
    exit(1)

# Save the output
print(f"\nSaving image to: {args.output}")
try:
    result_img.save(args.output)
    print("✓ Image saved successfully")
except Exception as e:
    print(f"✗ Error saving image: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Generation completed successfully!")
print("=" * 60)
print(f"\nGenerated image saved to: {args.output}")
