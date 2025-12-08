from text_parser.text_parser_impl_2 import TextParserImpl2
from text_to_3d.shap_e import ShapE
from scene_composition.two_viewpoints_scene import TwoViewpointsScene
from depth_extraction.two_viewpoints_depth_extractor import TwoViewpointsDepthExtractor
import os
import argparse
import torch
from pytorch3d.structures import join_meshes_as_scene

def run_two_viewpoints_pipeline(
    prompt,
    output_dir="results",
    camera_distance=5.0,
    camera_elevation=25.0,
    camera_azimuth=10.0
):
    """
    Execute the two-viewpoints pipeline.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing prompt: {prompt}")
    
    # 1. Parse Text
    text_parser = TextParserImpl2()
    obj_name_list = text_parser.parse(prompt)
    if not obj_name_list:
        print("No objects found in prompt.")
        return

    # 2. Generate Objects (Shap-E)
    # Using orientation from original script: [-90, 180, 0] to fix Shap-E defaults
    shap_e = ShapE(orientation=[-90.0, 180.0, 0.0])
    obj_paths = shap_e.convert_multiple_texts_to_3d(
        texts=obj_name_list,
        output_dir=f"{output_dir}/shap_e"
    )
    print(f"Generated objects: {obj_name_list}")

    # 3. Initialize Scene
    # Use parameters that position Mirror at Z = -3.0 (mirror_gap_ahead=3.0)
    mirror_gap_ahead = 3.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    scene_compositor = TwoViewpointsScene(
        gap=0.2,
        min_angle=-0.3, # Matching default
        max_angle=0.3,
        mirror_thickness=0.1,
        mirror_gap_side=2.0,
        mirror_gap_top=2.0,
        mirror_gap_ahead=mirror_gap_ahead,
        device=device
    )
    
    print("Composing scene...")
    # This creates the raw scene components including 'reflections' which we won't use
    scene = scene_compositor.compose_scene(obj_paths)
    
    # 4. Viewpoint 1: Camera behind object, slightly to left/right
    # User specified: "object, mirror frame and the floor"
    # We explicitly exclude 'reflections'
    
    view1_scene_components = {
        'objects': scene['objects'],
        'mirror': scene['mirror'],
        'floor': scene['floor'],
        'reflections': [] # Explicitly empty
    }

    print("Extracting Depth Map 1 (Direct View)...")
    extractor = TwoViewpointsDepthExtractor(
        image_size=(1024, 1024),
        output_dir=f"{output_dir}/depth",
        device=device,
        fov=60.0
    )
    
    # Extract View 1
    dm1 = extractor.extract_depth_map(
        view1_scene_components,
        output_prefix="view1_direct",
        camera_params=(camera_distance, camera_elevation, camera_azimuth)
    )
    print(f"  Saved View 1 to: {dm1.image_path}")

    # 5. Flip Camera and Object for View 2
    print("Preparing View 2 (Mirror View)...")
    
    # A. Flip Objects
    # We update the 'objects' list in the scene dictionary with flipped versions
    # User modified flip_object_z logic is in TwoViewpointsScene
    original_objects = scene['objects']
    flipped_objects = scene_compositor.flip_objects_z(original_objects)
    
    # Create a new scene dictionary for View 2
    # User specified: "just object and floor" (no mirror, no reflections)
    scene_view2 = {
        'objects': flipped_objects,
        'mirror': scene['mirror'], # No mirror frame
        'reflections': [], # No reflections
        'floor': scene['floor']
    }
    
    # B. Calculate Flipped Camera
    # Mirror is at Z = -mirror_gap_ahead.
    mirror_z = -mirror_gap_ahead
    
    camera_view2 = extractor.get_mirror_view_camera(
        mirror_z=mirror_z,
        original_dist=camera_distance,
        original_elev=camera_elevation,
        original_azim=camera_azimuth
    )
    
    print("Extracting Depth Map 2 (Flipped View)...")
    dm2 = extractor.extract_depth_map(
        scene_view2,
        output_prefix="view2_flipped",
        cameras=camera_view2
    )
    print(f"  Saved View 2 to: {dm2.image_path}")
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="a chair")
    parser.add_argument('--output-dir', type=str, default="results/two_view_test")
    parser.add_argument('--camera-distance', type=float, default=5.0)
    parser.add_argument('--camera-elevation', type=float, default=25.0)
    parser.add_argument('--camera-azimuth', type=float, default=15.0) 
    
    args = parser.parse_args()
    
    run_two_viewpoints_pipeline(
        prompt=args.prompt,
        output_dir=args.output_dir,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        camera_azimuth=args.camera_azimuth
    )
