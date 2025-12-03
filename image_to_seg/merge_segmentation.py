import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n distinct colors using HSV color space for better visual separation.

    Args:
        n: Number of distinct colors to generate

    Returns:
        List of RGB tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB
        # Using full saturation and value for vibrant colors
        if hue < 1/6:
            r, g, b = 1, hue * 6, 0
        elif hue < 2/6:
            r, g, b = (2/6 - hue) * 6, 1, 0
        elif hue < 3/6:
            r, g, b = 0, 1, (hue - 2/6) * 6
        elif hue < 4/6:
            r, g, b = 0, (4/6 - hue) * 6, 1
        elif hue < 5/6:
            r, g, b = (hue - 4/6) * 6, 0, 1
        else:
            r, g, b = 1, 0, (1 - hue) * 6

        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return colors


def merge_segmentation_maps(
    result_dir: str,
    output_path: str = None,
    output_json_path: str = None
) -> Dict[str, Dict[str, int]]:
    """
    Merge multiple binary segmentation masks into a single colored segmentation map.
    Each white segment from different masks gets a unique color.

    Args:
        result_dir: Directory containing the mask PNG files
        output_path: Path to save the merged segmentation map (optional)
        output_json_path: Path to save the color mapping JSON (optional)

    Returns:
        Dictionary mapping mask names to their RGB color values
    """
    result_path = Path(result_dir)

    # Find all mask files
    mask_files = sorted(result_path.glob("mask_*.png"))

    if not mask_files:
        raise ValueError(f"No mask files found in {result_dir}")

    print(f"Found {len(mask_files)} mask files")

    # Read the first mask to get dimensions
    first_mask = Image.open(mask_files[0])
    height, width = first_mask.size[1], first_mask.size[0]

    # Create the merged segmentation map (RGB)
    merged_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate distinct colors for each mask
    colors = generate_distinct_colors(len(mask_files))

    # Dictionary to store mask name -> RGB color mapping
    color_mapping = {}

    # Process each mask
    for idx, mask_file in enumerate(mask_files):
        mask_name = mask_file.stem  # Get filename without extension
        print(f"Processing {mask_name}...")

        # Load mask as grayscale
        mask = Image.open(mask_file).convert('L')
        mask_array = np.array(mask)

        # Find white pixels (assuming white is > 127)
        white_pixels = mask_array > 127

        # Assign the unique color to this mask's white pixels
        rgb_color = colors[idx]
        merged_map[white_pixels] = rgb_color

        # Store in color mapping
        color_mapping[mask_name] = {
            "r": rgb_color[0],
            "g": rgb_color[1],
            "b": rgb_color[2]
        }

    # Save the merged segmentation map if output path is provided
    if output_path:
        merged_image = Image.fromarray(merged_map)
        merged_image.save(output_path)
        print(f"\nMerged segmentation map saved to {output_path}")

    # Save the color mapping JSON if output path is provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(color_mapping, f, indent=2)
        print(f"Color mapping JSON saved to {output_json_path}")

    return color_mapping


if __name__ == "__main__":
    # Example usage
    result_dir = "segmentation_result/result3"
    output_image_path = "merge_segment_result/segmentation_3.png"
    output_json_path = "merge_segment_result/color_mapping_3.json"

    color_mapping = merge_segmentation_maps(
        result_dir=result_dir,
        output_path=output_image_path,
        output_json_path=output_json_path
    )

    print("\n=== Color Mapping ===")
    print(json.dumps(color_mapping, indent=2))
