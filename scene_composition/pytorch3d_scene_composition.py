import torch
from pytorch3d.io import load_objs_as_meshes, load_obj, IO
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from typing import List, Dict, Tuple
import random
import numpy as np
from pathlib import Path


class SceneComposition:
    """
    PyTorch3D-based scene composition that creates a scene with objects and their mirror reflections.
    Returns the scene structure instead of saving to file.
    """

    def __init__(
        self,
        gap: float = 0.1,
        min_angle: float = -np.pi/4,
        max_angle: float = np.pi/4,
        mirror_thickness: float = 0.1,
        mirror_gap_side: float = 2,
        mirror_gap_top: float = 2,
        mirror_gap_ahead: float = 3,
        device: str = "cpu"
    ):
        """
        Args:
            gap: Distance between objects horizontally
            min_angle: Minimum rotation angle around Y-axis (in radians)
            max_angle: Maximum rotation angle around Y-axis (in radians)
            mirror_thickness: Thickness of the mirror frame border
            mirror_gap_side: Horizontal margin on each side of the mirror
            mirror_gap_top: Vertical margin at the top of the mirror
            mirror_gap_ahead: Distance from objects to mirror plane
            device: Device to use for PyTorch tensors ('cpu' or 'cuda')
        """
        self.gap = gap
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.mirror_thickness = mirror_thickness
        self.mirror_gap_side = mirror_gap_side
        self.mirror_gap_top = mirror_gap_top
        self.mirror_gap_ahead = mirror_gap_ahead
        self.device = torch.device(device)

    def _load_objects(self, object_paths: List[str]) -> List[Meshes]:
        """Load meshes from file paths using PyTorch3D."""
        meshes = []
        io = IO()

        for object_path in object_paths:
            path = Path(object_path)

            # PyTorch3D supports different formats
            if path.suffix.lower() in ['.obj']:
                # Load OBJ file
                verts, faces, aux = load_obj(object_path, device=self.device)
                mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
            elif path.suffix.lower() in ['.ply']:
                # Load PLY file
                mesh = io.load_mesh(object_path, device=self.device)
            else:
                # Try generic load - may need trimesh conversion for GLB
                # For GLB files, we'll need to convert via trimesh
                import trimesh
                trimesh_mesh = trimesh.load(object_path, force='mesh')

                # Convert trimesh to pytorch3d
                verts = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32, device=self.device)
                faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64, device=self.device)
                mesh = Meshes(verts=[verts], faces=[faces])

            meshes.append(mesh)

        return meshes

    def _get_mesh_bounds(self, mesh: Meshes) -> torch.Tensor:
        """Get bounding box of a mesh. Returns [2, 3] tensor with min and max corners."""
        verts = mesh.verts_packed()
        min_bounds = verts.min(dim=0)[0]
        max_bounds = verts.max(dim=0)[0]
        return torch.stack([min_bounds, max_bounds])

    def _get_mesh_centroid(self, mesh: Meshes) -> torch.Tensor:
        """Get centroid of a mesh."""
        verts = mesh.verts_packed()
        return verts.mean(dim=0)

    def _apply_translation(self, mesh: Meshes, translation: torch.Tensor) -> Meshes:
        """Apply translation to a mesh."""
        verts = mesh.verts_packed()
        new_verts = verts + translation
        return Meshes(verts=[new_verts], faces=mesh.faces_list())

    def _apply_transform(self, mesh: Meshes, transform_matrix: torch.Tensor) -> Meshes:
        """Apply 4x4 transformation matrix to a mesh."""
        verts = mesh.verts_packed()
        # Convert to homogeneous coordinates
        ones = torch.ones((verts.shape[0], 1), device=self.device)
        verts_homo = torch.cat([verts, ones], dim=1)
        # Apply transformation
        new_verts_homo = torch.matmul(verts_homo, transform_matrix.t())
        # Convert back to 3D
        new_verts = new_verts_homo[:, :3]
        return Meshes(verts=[new_verts], faces=mesh.faces_list())

    def _put_single_object_on_plane_xz(self, mesh: Meshes) -> Meshes:
        """Place object on XZ plane by setting its minimum Y to 0."""
        bounds = self._get_mesh_bounds(mesh)
        min_y = bounds[0, 1]
        translation = torch.tensor([0.0, -min_y, 0.0], device=self.device)
        return self._apply_translation(mesh, translation)

    def _put_objects_on_plane_xz(self, meshes: List[Meshes]) -> List[Meshes]:
        """Put all objects on the XZ plane."""
        return [self._put_single_object_on_plane_xz(mesh) for mesh in meshes]

    def _add_random_rotation(self, meshes: List[Meshes]) -> List[Meshes]:
        """Add random rotation around Y-axis to each object."""
        rotated_meshes = []

        for mesh in meshes:
            # Random angle
            angle = random.uniform(self.min_angle, self.max_angle)

            # Get centroid
            centroid = self._get_mesh_centroid(mesh)

            # Create rotation matrix around Y-axis
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # 3x3 rotation matrix around Y-axis
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=torch.float32, device=self.device)

            # Translate to origin, rotate, translate back
            verts = mesh.verts_packed()
            centered_verts = verts - centroid
            rotated_verts = torch.matmul(centered_verts, R.t())
            final_verts = rotated_verts + centroid

            rotated_mesh = Meshes(verts=[final_verts], faces=mesh.faces_list())
            rotated_meshes.append(rotated_mesh)

        return rotated_meshes

    def _shift_objects_to_center(self, meshes: List[Meshes]) -> List[Meshes]:
        """Center objects horizontally around X=0."""
        if len(meshes) == 0:
            return meshes

        first_bounds = self._get_mesh_bounds(meshes[0])
        last_bounds = self._get_mesh_bounds(meshes[-1])

        min_x = first_bounds[0, 0]
        max_x = last_bounds[1, 0]
        shift_x = (min_x + max_x) / 2

        translation = torch.tensor([-shift_x, 0, 0], dtype=torch.float32, device=self.device)

        return [self._apply_translation(mesh, translation) for mesh in meshes]

    def _put_objects_next_to_each_other(self, meshes: List[Meshes]) -> List[Meshes]:
        """Arrange objects horizontally with gaps between them."""
        if len(meshes) < 2:
            return meshes

        positioned_meshes = [meshes[0]]
        previous_mesh = meshes[0]

        for mesh in meshes[1:]:
            prev_bounds = self._get_mesh_bounds(previous_mesh)
            curr_bounds = self._get_mesh_bounds(mesh)

            # Calculate shift to place next to previous object with gap
            shift_x = prev_bounds[1, 0] - curr_bounds[0, 0] + self.gap
            translation = torch.tensor([shift_x, 0, 0], dtype=torch.float32, device=self.device)

            positioned_mesh = self._apply_translation(mesh, translation)
            positioned_meshes.append(positioned_mesh)
            previous_mesh = positioned_mesh

        # Center all objects
        return self._shift_objects_to_center(positioned_meshes)

    def _total_objects_width(self, meshes: List[Meshes]) -> float:
        """Calculate total width of all objects."""
        if len(meshes) == 0:
            return 0

        first_bounds = self._get_mesh_bounds(meshes[0])
        last_bounds = self._get_mesh_bounds(meshes[-1])

        width = (last_bounds[1, 0] - first_bounds[0, 0]).item()
        return width

    def _max_objects_height(self, meshes: List[Meshes]) -> float:
        """Get maximum height among all objects."""
        if len(meshes) == 0:
            return 0

        max_height = 0
        for mesh in meshes:
            bounds = self._get_mesh_bounds(mesh)
            height = (bounds[1, 1] - bounds[0, 1]).item()
            max_height = max(max_height, height)

        return max_height

    def _max_objects_depth(self, meshes: List[Meshes]) -> float:
        """Get maximum depth among all objects."""
        if len(meshes) == 0:
            return 0

        max_depth = 0
        for mesh in meshes:
            bounds = self._get_mesh_bounds(mesh)
            depth = (bounds[1, 2] - bounds[0, 2]).item()
            max_depth = max(max_depth, depth)

        return max_depth

    def _create_box_mesh(self, width: float, height: float, depth: float) -> Meshes:
        """Create a box mesh with given dimensions."""
        # Create vertices for a box centered at origin
        w, h, d = width / 2, height / 2, depth / 2

        verts = torch.tensor([
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],  # Front face
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],       # Back face
        ], dtype=torch.float32, device=self.device)

        # Define faces (12 triangles for 6 faces of the box)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # Front
            [4, 6, 5], [4, 7, 6],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
            [3, 2, 6], [3, 6, 7],  # Top
            [0, 4, 5], [0, 5, 1],  # Bottom
        ], dtype=torch.int64, device=self.device)

        return Meshes(verts=[verts], faces=[faces])

    def _make_mirror_frame(self, outer_width: float, outer_height: float, thickness: float, depth: float = 0.05) -> Meshes:
        """
        Create a hollow rectangular mirror frame.
        Note: PyTorch3D doesn't have built-in boolean operations like trimesh,
        so we'll create the frame geometry manually.
        """
        # We'll create the frame as 4 rectangular bars forming a border
        # Top, Bottom, Left, Right

        # Dimensions
        inner_width = outer_width - 2 * thickness
        inner_height = outer_height - 2 * thickness

        all_verts = []
        all_faces = []
        vertex_offset = 0

        # Create 4 bars: top, bottom, left, right
        # Top bar
        top_bar = self._create_box_mesh(outer_width, thickness, depth)
        top_verts = top_bar.verts_packed() + torch.tensor([0, outer_height/2 - thickness/2, 0], device=self.device)
        all_verts.append(top_verts)
        all_faces.append(top_bar.faces_packed())
        vertex_offset += top_verts.shape[0]

        # Bottom bar
        bottom_bar = self._create_box_mesh(outer_width, thickness, depth)
        bottom_verts = bottom_bar.verts_packed() + torch.tensor([0, -outer_height/2 + thickness/2, 0], device=self.device)
        all_verts.append(bottom_verts)
        all_faces.append(bottom_bar.faces_packed() + vertex_offset)
        vertex_offset += bottom_verts.shape[0]

        # Left bar
        left_bar = self._create_box_mesh(thickness, inner_height, depth)
        left_verts = left_bar.verts_packed() + torch.tensor([-outer_width/2 + thickness/2, 0, 0], device=self.device)
        all_verts.append(left_verts)
        all_faces.append(left_bar.faces_packed() + vertex_offset)
        vertex_offset += left_verts.shape[0]

        # Right bar
        right_bar = self._create_box_mesh(thickness, inner_height, depth)
        right_verts = right_bar.verts_packed() + torch.tensor([outer_width/2 - thickness/2, 0, 0], device=self.device)
        all_verts.append(right_verts)
        all_faces.append(right_bar.faces_packed() + vertex_offset)

        # Combine all vertices and faces
        combined_verts = torch.cat(all_verts, dim=0)
        combined_faces = torch.cat(all_faces, dim=0)

        frame = Meshes(verts=[combined_verts], faces=[combined_faces])

        # Put frame on XZ plane
        frame = self._put_single_object_on_plane_xz(frame)

        return frame

    def _create_mirror_frame(self, meshes: List[Meshes]) -> Meshes:
        """Create mirror frame sized to fit all objects."""
        total_width = self._total_objects_width(meshes) + self.mirror_gap_side * 2
        total_height = self._max_objects_height(meshes) + self.mirror_gap_top

        mirror_frame = self._make_mirror_frame(total_width, total_height, self.mirror_thickness)

        # Move mirror behind objects
        translation = torch.tensor([0, 0, -self.mirror_gap_ahead], dtype=torch.float32, device=self.device)
        mirror_frame = self._apply_translation(mirror_frame, translation)

        return mirror_frame

    def _calculate_objects_reflection(self, meshes: List[Meshes]) -> List[Meshes]:
        """Create mirror reflections of objects."""
        reflected_meshes = []

        for mesh in meshes:
            # Copy the mesh
            verts = mesh.verts_packed().clone()
            faces = mesh.faces_list()

            # Reflect along Z-axis (negate Z coordinates)
            reflection_matrix = torch.diag(torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device))
            reflected_verts = torch.matmul(verts, reflection_matrix.t())

            # Move behind the mirror
            translation = torch.tensor([0, 0, -2 * self.mirror_gap_ahead], dtype=torch.float32, device=self.device)
            reflected_verts = reflected_verts + translation

            reflected_mesh = Meshes(verts=[reflected_verts], faces=faces)
            reflected_meshes.append(reflected_mesh)

        return reflected_meshes

    def compose_scene(self, object_paths: List[str]) -> Dict[str, List[Meshes]]:
        """
        Compose a scene with objects, mirror frame, and reflections.

        Args:
            object_paths: List of file paths to 3D object files

        Returns:
            Dictionary containing:
                - 'objects': List of original object meshes
                - 'mirror': List containing the mirror frame mesh
                - 'reflections': List of reflected object meshes
        """
        # Load and process objects
        objects = self._load_objects(object_paths)
        objects = self._put_objects_on_plane_xz(objects)
        objects = self._add_random_rotation(objects)
        objects = self._put_objects_next_to_each_other(objects)

        # Create mirror frame
        mirror_frame = self._create_mirror_frame(objects)

        # Create reflections
        reflections = self._calculate_objects_reflection(objects)

        # Return scene as structured dictionary
        scene = {
            'objects': objects,
            'mirror': [mirror_frame],
            'reflections': reflections
        }

        return scene


if __name__ == "__main__":
    # Example usage
    scene_composition = SceneComposition(device="cpu")
    scene = scene_composition.compose_scene([
        "dog_ahead.glb",
        "cat_ahead.glb",
        "lamp.glb",
        "chair_ahead.glb"
    ])

    print(f"Scene composition complete!")
    print(f"Objects: {len(scene['objects'])}")
    print(f"Mirror frames: {len(scene['mirror'])}")
    print(f"Reflections: {len(scene['reflections'])}")
