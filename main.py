from text_parser.text_parser_impl import TextParserImpl
from text_to_3d.shap_e import ShapE
from scene_composition.pytorch3d_scene_composition import SceneComposition
from condition_extraction import PyTorch3DConditionExtractor
import os

os.makedirs("results", exist_ok=True)

text_parser = TextParserImpl()
text = "a teddy bear in front of the mirror"

obj_name_list = text_parser.parse(text)

shap_e = ShapE(orientation=[-90.0, 180.0, 0.0])
obj_paths = shap_e.convert_multiple_texts_to_3d(
    texts=obj_name_list,
    output_dir="results/shap_e"
)

print("Objects are created successfully by Shap-E")

print("=" * 60)
print("PyTorch3D Scene Composition & Condition Extraction")
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

print("\nExtracting segmentation map and metadata...")
extractor = PyTorch3DConditionExtractor(
    image_size=(1024, 1024),
    output_dir="results/segmentation",
    device="cuda",
    camera_distance=5.0,
    camera_elevation=25.0,
    camera_azimuth=10.0,
    fov=60.0,
    faces_per_pixel=1,
    segment_descriptions={
        'objects': 'A real object standing in front of the mirror',
        'mirror': 'The mirror frame surrounding the reflective surface',
        'reflections': 'A reflection of the object visible inside the mirror'
    },
    caption=None,  # Let it auto-generate from object names
    seed=42
)

condition_map = extractor.extract_condition_map(
    scene,
    output_prefix="scene_segmentation",
    object_paths=obj_paths
)

print(f"  ✓ Segmentation map saved to: {condition_map.image_path}")
print(f"  ✓ Metadata JSON saved to: {condition_map.json_path}")
