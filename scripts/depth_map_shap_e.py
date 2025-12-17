from camera.interpolation import interpolate_view_params
import random
from text_parser.text_parser_impl_2 import TextParserImpl2
from text_to_3d.shap_e import ShapE
from scene_composition.pytorch3d_scene_composition import SceneComposition
from depth_extraction.pytorch3d_depth_extractor import PyTorch3DDepthExtractor
import os
import argparse
from camera.camera_optimizer import optimize_view

def generate_depth_for_prompt(
    prompt,
    output_dir="results",
    mirror_gap_ahead=3.0,
    guidance_scale=15.0,
    karras_steps=128,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0
):
    """
    Generate a depth map for a given text prompt.
    
    Args:
        prompt: Text description of the scene (e.g., "a teddy bear in front of the mirror")
        output_dir: Base directory for all outputs
        mirror_gap_ahead: Distance from objects to mirror
        guidance_scale: Guidance scale for Shap-E
        karras_steps: Number of steps for Karras scheduler
        sigma_min: Minimum sigma for Karras scheduler
        sigma_max: Maximum sigma for Karras scheduler
        s_churn: Churn for Karras scheduler
    
    Returns:
        str: Path to the generated depth map image
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing prompt: {prompt}")
    print("=" * 60)
    
    # Parse the prompt to extract object names
    text_parser = TextParserImpl2()
    obj_name_list = text_parser.parse(prompt)
    
    # Generate 3D objects from text using Shap-E
    # Generate 3D objects from text using Shap-E
    shap_e = ShapE(
        orientation=[-90.0, 180.0, 0.0],
        guidance=guidance_scale,
        karras_steps=karras_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        s_churn=s_churn
    )
    obj_paths = shap_e.convert_multiple_texts_to_3d(
        texts=obj_name_list,
        output_dir=f"{output_dir}/shap_e"
    )
    
    print("Objects are created successfully by Shap-E")
    
    print("=" * 60)
    print("PyTorch3D Scene Composition & Depth Extraction")
    print("=" * 60)
    
    # Compose the 3D scene with objects, mirror, and reflections
    print("\nComposing scene with objects and mirror...")
    scene_compositor = SceneComposition(
        gap=0.2,
        min_angle=-0.3,
        max_angle=0.3,
        mirror_thickness=0.1,
        mirror_gap_side=2.0,
        mirror_gap_top=2.0,
        mirror_gap_ahead=mirror_gap_ahead,
        device="cuda"
    )
    
    scene = scene_compositor.compose_scene(obj_paths)
    
    print(f"  ✓ Scene created with:")
    print(f"    - {len(scene['objects'])} objects")
    print(f"    - {len(scene['mirror'])} mirror frame")
    print(f"    - {len(scene['reflections'])} reflections")

    # Calculate camera components
    camera_distance = mirror_gap_ahead * 1.5
    camera_elevation = 25.0
    camera_azimuth = 0.0
    
    # Generate pool of 19 camera poses
    print("\nGenerating camera pose pool...")
    start_params = (camera_distance, camera_elevation, camera_azimuth - 20.0)
    end_params = (camera_distance, camera_elevation, camera_azimuth + 20.0)
    
    pose_pool = interpolate_view_params(start_params, end_params, num_steps=19)
    
    # Randomly select one pose
    selected_idx = random.randint(0, len(pose_pool) - 1)
    selected_pose = pose_pool[selected_idx]
    
    print(f"  ✓ Created pool of {len(pose_pool)} poses from {start_params} to {end_params}")
    print(f"  ✓ Randomly selected pose #{selected_idx}: {selected_pose}")
    
    # Update camera parameters with the selected pose
    dist, elev, azim = selected_pose

    # Extract depth map from the composed scene
    print("\nExtracting depth map...")
    extractor = PyTorch3DDepthExtractor(
        image_size=(512, 512),
        output_dir=f"{output_dir}/depth",
        device="cuda",
        # We can pass defaults here, but we will override in the method call
        camera_distance=dist,
        camera_elevation=elev,
        camera_azimuth=azim,
        fov=60.0,
        faces_per_pixel=1,
        normalize=True,
        invert=True
    )
    
    # Pass the optimized parameters explicitly (redundant if init updated, but safe)
    depth_map = extractor.extract_depth_map(
        scene,
        output_prefix="scene_depth_shap_e",
        object_paths=obj_paths,
        camera_params=(dist, elev, azim)
    )
    
    print(f"  ✓ Depth map saved to: {depth_map.image_path}")
    print("\nDepth extraction completed successfully!")
    
    return depth_map.image_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate 3D scene with mirror reflections and depth map from text prompt'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="a teddy bear in front of the mirror",
        help='Text prompt describing the scene (e.g., "a teddy bear in front of the mirror")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="results",
        help='Base output directory (default: results)'
    )
    
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=15.0,
        help='Guidance scale for Shap-E (default: 15.0)'
    )
    parser.add_argument(
        '--karras-steps',
        type=int,
        default=64,
        help='Number of steps for Karras scheduler (default: 64)'
    )
    parser.add_argument(
        '--sigma-min',
        type=float,
        default=1e-3,
        help='Minimum sigma for Karras scheduler (default: 1e-3)'
    )
    parser.add_argument(
        '--sigma-max',
        type=float,
        default=160,
        help='Maximum sigma for Karras scheduler (default: 160)'
    )
    parser.add_argument(
        '--s-churn',
        type=float,
        default=0,
        help='Churn for Karras scheduler (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Call the main function
    depth_map_path = generate_depth_for_prompt(
        prompt=args.prompt,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        karras_steps=args.karras_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn
    )
    
    print(f"\n{'=' * 60}")
    print(f"Final output: {depth_map_path}")
    print(f"{'=' * 60}")
