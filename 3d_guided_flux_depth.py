from text_parser.text_parser_impl_2 import TextParserImpl2
from text_to_3d.shap_e import ShapE
from scene_composition.pytorch3d_scene_composition import SceneComposition
from depth_extraction.pytorch3d_depth_extractor import PyTorch3DDepthExtractor
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate 3D scene with mirror reflections and depth map from text prompt')
parser.add_argument(
    '--prompt',
    type=str,
    default="a teddy bear in front of the mirror",
    help='Text prompt describing the scene (e.g., "a teddy bear in front of the mirror")'
)
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

text_parser = TextParserImpl2()
text = args.prompt

print(f"Processing prompt: {text}")
print("=" * 60)

obj_name_list = text_parser.parse(text)

shap_e = ShapE(orientation=[-90.0, 180.0, 0.0])
obj_paths = shap_e.convert_multiple_texts_to_3d(
    texts=obj_name_list,
    output_dir="results/shap_e"
)

print("Objects are created successfully by Shap-E")

print("=" * 60)
print("PyTorch3D Scene Composition & Depth Extraction")
print("=" * 60)

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

print("\nExtracting depth map...")
extractor = PyTorch3DDepthExtractor(
    image_size=(1024, 1024),
    output_dir="results/depth",
    device="cuda",
    camera_distance=5.0,
    camera_elevation=25.0,
    camera_azimuth=10.0,
    fov=60.0,
    faces_per_pixel=1,
    normalize=True,
    invert=True
)

depth_map = extractor.extract_depth_map(
    scene,
    output_prefix="scene_depth",
    object_paths=obj_paths
)

print(f"  ✓ Depth map saved to: {depth_map.image_path}")
print("\nDepth extraction completed successfully!")
