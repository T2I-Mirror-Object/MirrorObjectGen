from scene_composition.pytorch3d_scene_composition import SceneComposition
from pytorch3d.io import save_obj
from pytorch3d.structures import join_meshes_as_scene
import torch
import os

# Create output directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create scene composition
scene_composition = SceneComposition(device="cpu")
scene = scene_composition.compose_scene([
    "data/dog_ahead.glb",
    "data/cat_ahead.glb",
    "data/lamp.glb",
    "data/chair_ahead.glb"
])

# Option 1: Save all components as separate OBJ files
print("Saving scene components as separate OBJ files...")

all_meshes = scene['objects'] + scene['mirror'] + scene['reflections']

# Combine all vertices and faces with proper offset
all_verts = []
all_faces = []
vertex_offset = 0

for mesh in all_meshes:
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    all_verts.append(verts)
    all_faces.append(faces + vertex_offset)
    vertex_offset += verts.shape[0]

# Concatenate all
combined_verts = torch.cat(all_verts, dim=0)
combined_faces = torch.cat(all_faces, dim=0)

# Save combined scene
save_obj("results/scene_combined.obj", combined_verts, combined_faces)
print(f"  Saved combined scene to results/scene_combined.obj")

print("\nScene composition complete!")
print(f"Total objects: {len(scene['objects'])}")
print(f"Mirror frames: {len(scene['mirror'])}")
print(f"Reflections: {len(scene['reflections'])}")