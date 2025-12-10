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
    camera_distance=5.0,
    camera_elevation=25.0,
    camera_azimuth=10.0
):
    """
    Generate a depth map for a given text prompt.
    
    Args:
        prompt: Text description of the scene (e.g., "a teddy bear in front of the mirror")
        output_dir: Base directory for all outputs (default: "results")
        camera_distance: Initial guess for distance (default: 5.0)
        camera_elevation: Initial guess for elevation (default: 25.0)
        camera_azimuth: Initial guess for azimuth (default: 10.0)
    
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
    shap_e = ShapE(orientation=[-90.0, 180.0, 0.0])
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
        mirror_gap_ahead=3.0,
        device="cuda"
    )
    
    scene = scene_compositor.compose_scene(obj_paths)
    
    print(f"  ✓ Scene created with:")
    print(f"    - {len(scene['objects'])} objects")
    print(f"    - {len(scene['mirror'])} mirror frame")
    print(f"    - {len(scene['reflections'])} reflections")
    
    # Optimize Camera View
    print("\nOptimizing Camera View...")
    opt_dist, opt_elev, opt_azim = optimize_view(scene, device="cuda")
    
    print(f"  ✓ Optimized View: Dist={opt_dist:.2f}, Elev={opt_elev:.2f}, Azim={opt_azim:.2f}")

    # Extract depth map from the composed scene
    print("\nExtracting depth map...")
    extractor = PyTorch3DDepthExtractor(
        image_size=(1024, 1024),
        output_dir=f"{output_dir}/depth",
        device="cuda",
        # We can pass defaults here, but we will override in the method call
        camera_distance=opt_dist,
        camera_elevation=opt_elev,
        camera_azimuth=opt_azim,
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
        camera_params=(opt_dist, opt_elev, opt_azim)
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
        '--camera-distance',
        type=float,
        default=5.0,
        help='Distance of camera from origin (default: 5.0)'
    )
    parser.add_argument(
        '--camera-elevation',
        type=float,
        default=25.0,
        help='Camera elevation angle in degrees (default: 25.0)'
    )
    parser.add_argument(
        '--camera-azimuth',
        type=float,
        default=10.0,
        help='Camera azimuth angle in degrees (default: 10.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="results",
        help='Base output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Call the main function
    depth_map_path = generate_depth_for_prompt(
        prompt=args.prompt,
        output_dir=args.output_dir,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        camera_azimuth=args.camera_azimuth
    )
    
    print(f"\n{'=' * 60}")
    print(f"Final output: {depth_map_path}")
    print(f"{'=' * 60}")
