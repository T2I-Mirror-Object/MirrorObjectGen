import torch
from typing import Tuple, Optional, Dict, List
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import CamerasBase, FoVPerspectiveCameras
from pytorch3d.renderer import look_at_view_transform

from depth_extraction.pytorch3d_depth_extractor import PyTorch3DDepthExtractor

class TwoViewpointsDepthExtractor(PyTorch3DDepthExtractor):
    """
    Depth extractor that supports calculating a flipped camera view across a mirror plane.
    """

    def get_mirror_view_camera(
        self, 
        mirror_z: float,
        original_dist: float, 
        original_elev: float, 
        original_azim: float
    ) -> CamerasBase:
        """
        Calculate the camera reflected across the mirror plane.
        
        The camera is originally positioned using look_at_view_transform(dist, elev, azim).
        We calculate the position of this camera, reflect it across the plane Z = mirror_z,
        and create a new camera at that position looking towards the center of the mirror (or object).
        
        Args:
            mirror_z: Z-coordinate of the mirror plane.
            original_dist: Distance of original camera.
            original_elev: Elevation of original camera.
            original_azim: Azimuth of original camera.
            
        Returns:
             A new PyTorch3D camera object.
        """
        
        # 1. Get original camera position (assuming looking at 0,0,0)
        # We need the Cartesian coordinates.
        # look_at_view_transform returns R (rotation) and T (translation) for the camera *view transform* (World->View).
        # To get the Camera Position in World Coordinates, we can inversely use the spherical coordinates logic 
        # or invert the T/R. PyTorch3D Camera Position C = -R^T * T.
        
        R, T = look_at_view_transform(dist=original_dist, elev=original_elev, azim=original_azim)
        
        # Calculate Camera Position in World Space
        # C = - R^T * T (Note: T in PyTorch3D is translation of the world relative to camera)
        C = -torch.bmm(R.transpose(1, 2), T[:, :, None])[:, :, 0] # Shape (1, 3)
        cam_pos_world = C[0] # [x, y, z]
        
        # 2. Reflect Camera Position across plane Z = mirror_z
        # P' = (x, y, 2*mirror_z - z)
        reflected_pos = cam_pos_world.clone()
        reflected_pos[2] = 2 * mirror_z - cam_pos_world[2]
        
        # 3. Determine 'At' point (Look At)
        # Original camera looks at (0.0, 0.0, 0.0) by default in look_at_view_transform.
        # The reflected camera should also look at (0.0, 0.0, 0.0) from the other side.
        # (Assuming we want to view the object at the origin)
        at = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        
        # 4. Create new LookAt Transform
        # We compute R, T for a camera at `reflected_pos` looking at `at`.
        # PyTorch3D `look_at_view_transform` can accept `eye` and `at`.
        
        eye = reflected_pos.unsqueeze(0) # (1, 3)
        
        R_new, T_new = look_at_view_transform(
            eye=eye,
            at=at,
            device=self.device
        )
        
        # 5. Create Camera
        camera = FoVPerspectiveCameras(
            device=self.device,
            R=R_new,
            T=T_new,
            fov=self.fov
        )
        
        return camera
