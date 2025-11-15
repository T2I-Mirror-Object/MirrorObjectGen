import trimesh
from trimesh.scene import Scene
from typing import List
import random
import numpy as np

class SceneComposition:
    def __init__(
        self, 
        gap: float = 0.1, 
        min_angle: float = -np.pi/4,
        max_angle: float = np.pi/4,
        mirror_thickness: float = 0.1,
        mirror_gap_side: float = 2,
        mirror_gap_top: float = 2,
        mirror_gap_ahead: float = 3,

    ):
        self.gap = gap
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.mirror_thickness = mirror_thickness
        self.mirror_gap_side = mirror_gap_side
        self.mirror_gap_top = mirror_gap_top
        self.mirror_gap_ahead = mirror_gap_ahead

    def _load_objects(self, object_paths: List[str]) -> List[trimesh.Trimesh]:
        objects = []
        for object_path in object_paths:
            object = trimesh.load(object_path)
            objects.append(object)
        return objects

    def _put_single_object_on_plane_xz(self, object: trimesh.Trimesh) -> trimesh.Trimesh:
        bound = object.bounds
        min_y = bound[0][1]
        translation = [0.0, -min_y, 0.0]
        object.apply_translation(translation)
        return object

    def _put_objects_on_plane_xz(self, objects: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        # put the bottom surface of the object on plane xz
        for object in objects:
            object = self._put_single_object_on_plane_xz(object)
        return objects

    def _add_random_rotation(self, objects: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        for object in objects:
            # in radians
            angle = random.uniform(self.min_angle, self.max_angle)
            center = object.centroid
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0], center)
            object.apply_transform(R)
        return objects
    
    def _shift_objects_to_center(self, objects: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        if len(objects) == 0:
            return objects
        first_object = objects[0]
        last_object = objects[-1]
        min_x = first_object.bounds[0, 0]
        max_x = last_object.bounds[1, 0]
        shift_x = (min_x + max_x) / 2
        translation = [-shift_x, 0, 0]
        for object in objects:
            object.apply_translation(translation)
        return objects

    def _put_objects_next_to_each_other(self, objects: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        if len(objects) < 2:
            return objects
        previous_object = objects[0]
        for object in objects[1:]:
            prev_bound = previous_object.bounds
            curr_bound = object.bounds
            gap = 0.1
            shift_x = prev_bound[1, 0] - curr_bound[0, 0] + gap
            translation = [shift_x, 0, 0]
            object.apply_translation(translation)
            previous_object = object
        
        # shift the objects to the center of the scene
        objects = self._shift_objects_to_center(objects)

        return objects

    def _total_objects_width(self, objects: List[trimesh.Trimesh]) -> float:
        if len(objects) == 0:
            return 0
        first_object = objects[0]
        last_object = objects[-1]
        width = last_object.bounds[1, 0] - first_object.bounds[0, 0]
        return width

    def _max_objects_height(self, objects: List[trimesh.Trimesh]) -> float:
        if len(objects) == 0:
            return 0
        max_height = 0
        for object in objects:
            height = object.bounds[1, 1] - object.bounds[0, 1]
            max_height = max(max_height, height)
        return max_height
    
    def _max_objects_depth(self, objects: List[trimesh.Trimesh]) -> float:
        if len(objects) == 0:
            return 0
        max_depth = 0
        for object in objects:
            depth = object.bounds[1, 2] - object.bounds[0, 2]
            max_depth = max(max_depth, depth)
        return max_depth

    def _make_mirror_frame(self, outer_width: float, outer_height: float, thickness: float, depth: float = 0.05) -> trimesh.Trimesh:
        outer = trimesh.creation.box(extents=[outer_width, outer_height, depth])
        inner_width = outer_width - 2 * thickness
        inner_height = outer_height - 2 * thickness
        inner = trimesh.creation.box(extents=[inner_width, inner_height, depth * 1.1])

        # Move inner box slightly forward so boolean works better
        inner.apply_translation([0, 0.001, 0])

        # Subtract inner from outer to get frame
        frame = outer.difference(inner)
        frame = self._put_single_object_on_plane_xz(frame)

        return frame

    def _create_mirror_frame(self, objects: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        total_width = self._total_objects_width(objects) + self.mirror_gap_side * 2
        total_height = self._max_objects_height(objects) + self.mirror_gap_top
        
        mirror_frame = self._make_mirror_frame(total_width, total_height, self.mirror_thickness)
        mirror_frame.apply_translation([0, 0, -self.mirror_gap_ahead])
        return mirror_frame

    def _calculate_objects_reflection(self, objects: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        copied_objects = []
        for obj in objects:
            reflected = obj.copy()

            # Step 1: Mirror along Z-axis (reflect across XZ plane)
            reflection_matrix = np.diag([1, 1, -1, 1])
            reflected.apply_transform(reflection_matrix)

            # Step 2: Move the reflected object behind the mirror
            translation = [0, 0, -2 * self.mirror_gap_ahead]
            reflected.apply_translation(translation)

            copied_objects.append(reflected)

        return copied_objects

    def compose_scene(self, object_paths: List[str]) -> trimesh.Trimesh:
        objects = self._load_objects(object_paths)
        objects = self._put_objects_on_plane_xz(objects)
        objects = self._add_random_rotation(objects)
        objects = self._put_objects_next_to_each_other(objects)
        mirror_frame = self._create_mirror_frame(objects)
        copied_objects = self._calculate_objects_reflection(objects)
        scene = Scene(objects + [mirror_frame] + copied_objects)
        return scene

    def export_scene(self, scene: trimesh.Trimesh, output_path: str):
        scene.export(output_path)
    

if __name__ == "__main__":
    scene_composition = SceneComposition()
    scene = scene_composition.compose_scene(["dog_ahead.glb", "cat_ahead.glb", "lamp.glb", "chair_ahead.glb"])
    scene_composition.export_scene(scene, "h2.glb")