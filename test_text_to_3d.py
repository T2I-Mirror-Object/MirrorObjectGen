#!/usr/bin/env python3
"""
Test script for InstantMesh Text-to-3D pipeline.
This script tests the text-to-3D conversion by generating 3D objects from text prompts.
"""

import os
import sys
import argparse
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_to_3d.instant_mesh import InstantMeshTextTo3D


def test_single_text(config_path: str, text: str, output_path: str = "test_outputs/"):
    """
    Test text-to-3D conversion with a single text prompt.

    Args:
        config_path: Path to InstantMesh config file
        text: Text description of the object
        output_path: Output directory for generated files
    """
    print("=" * 80)
    print("Testing Single Text to 3D Conversion")
    print("=" * 80)

    # Initialize the pipeline
    print(f"\nInitializing InstantMesh Text-to-3D pipeline...")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_path}")

    text_to_3d = InstantMeshTextTo3D(
        config_path=config_path,
        output_path=output_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        diffusion_steps=75,
        scale=1.0,
        view=6,
        export_texmap=False,
        no_rembg=False,
    )

    # Convert text to 3D
    print(f"\n{'=' * 80}")
    print(f"Converting text to 3D: \"{text}\"")
    print(f"{'=' * 80}\n")

    mesh_path = text_to_3d.convert_text_to_3d(text)

    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print(f"3D mesh generated at: {mesh_path}")
    print(f"{'=' * 80}\n")

    return mesh_path


def test_multiple_texts(config_path: str, texts: list, output_path: str = "test_outputs/"):
    """
    Test text-to-3D conversion with multiple text prompts.

    Args:
        config_path: Path to InstantMesh config file
        texts: List of text descriptions
        output_path: Output directory for generated files
    """
    print("=" * 80)
    print("Testing Multiple Texts to 3D Conversion")
    print("=" * 80)

    # Initialize the pipeline
    print(f"\nInitializing InstantMesh Text-to-3D pipeline...")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_path}")
    print(f"Number of texts: {len(texts)}")

    text_to_3d = InstantMeshTextTo3D(
        config_path=config_path,
        output_path=output_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        diffusion_steps=75,
        scale=1.0,
        view=6,
        export_texmap=False,
        no_rembg=False,
    )

    # Convert multiple texts to 3D
    print(f"\n{'=' * 80}")
    print(f"Converting {len(texts)} texts to 3D objects...")
    print(f"{'=' * 80}\n")

    mesh_paths = text_to_3d.convert_multiple_texts_to_3d(texts, output_dir=output_path)

    print(f"\n{'=' * 80}")
    print("BATCH CONVERSION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nGenerated {len([p for p in mesh_paths if p is not None])} out of {len(texts)} meshes:")
    for i, (text, path) in enumerate(zip(texts, mesh_paths)):
        status = "✓" if path else "✗"
        print(f"  {status} [{i+1}] \"{text}\"")
        if path:
            print(f"      -> {path}")
    print()

    return mesh_paths


def main():
    parser = argparse.ArgumentParser(
        description="Test InstantMesh Text-to-3D conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a single text prompt
  python test_text_to_3d.py --text "a red sports car"

  # Test with multiple predefined text prompts
  python test_text_to_3d.py --batch

  # Test with custom batch prompts
  python test_text_to_3d.py --batch --prompts "a red sports car" "a wooden chair" "a ceramic vase"

  # Use different config
  python test_text_to_3d.py --config InstantMesh/configs/instant-mesh-base.yaml --text "a wooden chair"

  # Custom output directory
  python test_text_to_3d.py --text "a blue teapot" --output my_outputs/
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='InstantMesh/configs/instant-mesh-large.yaml',
        help='Path to InstantMesh config file (default: instant-mesh-large.yaml)'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Single text prompt to convert to 3D'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch test with multiple text prompts'
    )

    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        help='List of text prompts for batch processing (use with --batch flag)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='test_outputs/',
        help='Output directory for generated files (default: test_outputs/)'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("\nAvailable configs in InstantMesh/configs/:")
        config_dir = os.path.join(os.path.dirname(__file__), "InstantMesh/configs")
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.yaml'):
                    print(f"  - InstantMesh/configs/{f}")
        sys.exit(1)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. Using CPU (this will be slow!)")

    print()

    # Run tests based on arguments
    if args.batch:
        # Use provided prompts or fall back to predefined test prompts
        if args.prompts:
            test_texts = args.prompts
        else:
            # Predefined test prompts
            test_texts = [
                "a red sports car",
                "a wooden chair",
                "a ceramic vase with flowers",
                "a modern lamp",
            ]
        test_multiple_texts(args.config, test_texts, args.output)

    elif args.text:
        # Single text test
        test_single_text(args.config, args.text, args.output)

    else:
        # No arguments provided, use default single test
        default_text = "a blue ceramic teapot"
        print("No text prompt provided. Using default test prompt.")
        print(f"(Use --text \"your prompt\" to specify a custom prompt)\n")
        test_single_text(args.config, default_text, args.output)


if __name__ == "__main__":
    main()
