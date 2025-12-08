import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate 2 images using FLUX.1-dev with ControlNet-Depth from 2 viewpoints')
parser.add_argument(
    '--prompt',
    type=str,
    required=True,
    help='Text prompt for image generation (used for both views)'
)
parser.add_argument(
    '--control-image-1',
    type=str,
    default="results/two_view_test/depth/view1_direct.png",
    help='Path to first control image (depth map view 1)'
)
parser.add_argument(
    '--control-image-2',
    type=str,
    default="results/two_view_test/depth/view2_flipped.png",
    help='Path to second control image (depth map view 2)'
)
parser.add_argument(
    '--output-dir',
    type=str,
    default="results/flux_controlnet_2views",
    help='Output directory for generated images'
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
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FLUX.1-ControlNet-Depth 2-Viewpoint Image Generation")
print("=" * 60)
print(f"\nPrompt: {args.prompt}")
print(f"Control Image 1: {args.control_image_1}")
print(f"Control Image 2: {args.control_image_2}")
print(f"Output Directory: {args.output_dir}")
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

def generate_and_save(control_image_path, output_filename):
    # Load control image
    print(f"\nLoading control image from: {control_image_path}")
    try:
        control_image = load_image(control_image_path)
        print(f"✓ Control image loaded: {control_image.size}")
    except Exception as e:
        print(f"✗ Error loading control image: {e}")
        return False

    # Generate image
    print(f"Generating image for {output_filename}...")
    
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
        return False

    # Save the output
    output_path = output_dir / output_filename
    print(f"Saving image to: {output_path}")
    try:
        image.save(output_path)
        print("✓ Image saved successfully")
        return True
    except Exception as e:
        print(f"✗ Error saving image: {e}")
        return False

# Generate for View 1
print("\n--- Processing View 1 ---")
success_1 = generate_and_save(args.control_image_1, "view1_generated.png")

# Generate for View 2
print("\n--- Processing View 2 ---")
# Use same generator/seed logic?
# Usually for consistency we might want to share seed, or use different if we want variations?
# User said "The 2 images use the same input prompt." and generally implied coherence.
# Same seed + different control image = usually preferred for consistency.
success_2 = generate_and_save(args.control_image_2, "view2_generated.png")

if success_1 and success_2:
    print("\n" + "=" * 60)
    print("Generation completed successfully for both views!")
    print("=" * 60)
    print(f"\nGenerated images saved to: {args.output_dir}")
else:
    print("\n" + "=" * 60)
    print("Generation finished with errors.")
    print("=" * 60)
    exit(1)
