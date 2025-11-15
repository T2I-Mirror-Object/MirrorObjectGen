from scene_composition import SceneComposition
from pytorch3d.io import save_glb

scene_composition = SceneComposition(device="cpu")
scene = scene_composition.compose_scene([
    "data/dog_ahead.glb",
    "data/cat_ahead.glb",
    "data/lamp.glb",
    "data/chair_ahead.glb"
])

# Save the scene to a glb file
save_glb(scene, "results/scene.glb")