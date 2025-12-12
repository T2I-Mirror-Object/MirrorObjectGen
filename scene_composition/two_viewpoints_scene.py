import torch
from pytorch3d.structures import Meshes
from scene_composition.pytorch3d_scene_composition import SceneComposition
from typing import List, Dict

class TwoViewpointsScene(SceneComposition):
    """
    Extended scene composition for two-viewpoint pipeline.
    Supports flipping objects across their own Z-axis.
    """

    def __init__(
        self,
        gap: float = 0.1,
        min_angle: float = -0.3, # Matching default in prompt's usage context if needed, but keeping class defaults
        max_angle: float = 0.3,
        mirror_thickness: float = 0.1,
        mirror_gap_side: float = 2,
        mirror_gap_top: float = 2,
        mirror_gap_ahead: float = 3,
        room_width: float = 20.0,
        room_depth: float = 20.0,
        wall_height: float = 10.0,
        wall_thickness: float = 0.5,
        device: str = "cpu"
    ):
        super().__init__(
            gap=gap,
            min_angle=min_angle,
            max_angle=max_angle,
            mirror_thickness=mirror_thickness,
            mirror_gap_side=mirror_gap_side,
            mirror_gap_top=mirror_gap_top,
            mirror_gap_ahead=mirror_gap_ahead,
            room_width=room_width,
            room_depth=room_depth,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            device=device
        )

    def flip_object_z(self, mesh: Meshes) -> Meshes:
        """
        Flip a mesh across its own Z-axis (local center).
        
        Args:
            mesh: Standard PyTorch3D Meshes object
            
        Returns:
            New Meshes object with flipped geometry
        """
        # Get centroid to define "own axis"
        centroid = self._get_mesh_centroid(mesh)
        center_z = centroid[2]
        
        # Get vertices
        verts = mesh.verts_packed().clone()
        
        reflection_matrix = torch.diag(torch.tensor([-1, 1, 1], dtype=torch.float32, device=self.device))
        reflected_verts = torch.matmul(verts, reflection_matrix.t())
        
        # Helper to reconstruct mesh
        new_mesh = Meshes(verts=[reflected_verts], faces=mesh.faces_list())
        
        return new_mesh

    def flip_objects_z(self, meshes: List[Meshes]) -> List[Meshes]:
        """Flip a list of meshes across each of their own Z-axes."""
        return [self.flip_object_z(mesh) for mesh in meshes]
