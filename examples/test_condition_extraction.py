from scene_composition import SceneComposition
from condition_extraction import PyTorch3DConditionExtractor
import os

# Create output directory
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("PyTorch3D Scene Composition & Condition Extraction Example")
print("=" * 60)

# Step 1: Compose the scene with objects and mirror
print("\n[Step 1] Composing scene with objects and mirror...")
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

object_paths = [
    "data/dog_ahead.glb",
    "data/cat_ahead.glb",
    "data/lamp.glb",
    "data/chair_ahead.glb"
]

scene = scene_compositor.compose_scene(object_paths)

print(f"  ✓ Scene created with:")
print(f"    - {len(scene['objects'])} objects")
print(f"    - {len(scene['mirror'])} mirror frame")
print(f"    - {len(scene['reflections'])} reflections")

# Step 2: Extract segmentation map and metadata
print("\n[Step 2] Extracting segmentation map and metadata...")
extractor = PyTorch3DConditionExtractor(
    image_size=(1024, 1024),
    output_dir="results",
    device="cuda",
    camera_distance=10.0,
    camera_elevation=0.0,
    camera_azimuth=20.0,
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
    object_paths=object_paths
)

print(f"  ✓ Segmentation map saved to: {condition_map.image_path}")
print(f"  ✓ Metadata JSON saved to: {condition_map.json_path}")

# Step 3: Display the JSON content
print("\n[Step 3] Segmentation metadata:")
import json
with open(condition_map.json_path, 'r') as f:
    metadata = json.load(f)

print(f"\n  Caption: {metadata['caption']}")
print(f"  Seed: {metadata.get('seed', 'N/A')}")
print(f"\n  Segments ({len(metadata['segments_info'])} total):")
for i, segment in enumerate(metadata['segments_info'], 1):
    color = segment['color']
    text = segment['text']
    print(f"    {i}. RGB{tuple(color)}: {text}")

print("\n" + "=" * 60)
print("Condition extraction complete!")
print("=" * 60)
print("\nOutput files:")
print(f"  - Segmentation image: {condition_map.image_path}")
print(f"  - Metadata JSON:      {condition_map.json_path}")
print("\nYou can now view the segmentation map to see the different")
print("segments color-coded, with each color corresponding to an entry")
print("in the JSON metadata file.")
