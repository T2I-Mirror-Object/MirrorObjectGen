import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRasterizer, # We only need the rasterizer, not the full renderer
    PointLights
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import interpolate_face_attributes # Crucial for normal map

from edge_extraction.edge_extractor import EdgeExtractor, EdgeMap

class PyTorch3DGeometricEdgeExtractor(EdgeExtractor):
    """
    Extracts high-quality edge maps using 3D Geometry (Depth + Normals) 
    instead of RGB processing.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        output_dir: str = "results/edge",
        device: str = "cpu",
        camera_distance: float = 5.0,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        fov: float = 60.0,
    ):
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.fov = fov

    def _create_camera(self) -> FoVPerspectiveCameras:
        R, T = look_at_view_transform(
            dist=self.camera_distance,
            elev=self.camera_elevation,
            azim=self.camera_azimuth,
        )
        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)

    def extract_edge_map(
        self,
        scene: Dict[str, List[Meshes]],
        output_prefix: str = "edge",
        camera_params: Optional[Tuple[float, float, float]] = None,
        cameras: Optional[FoVPerspectiveCameras] = None
    ) -> EdgeMap:
        
        # 1. Setup Camera
        if cameras is not None:
            camera = cameras
        elif camera_params is not None:
            dist, elev, azim = camera_params
            R, T = look_at_view_transform(dist, elev, azim)
            camera = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)
        else:
            camera = self._create_camera()

        # 2. Combine Meshes
        all_meshes = []
        for key in ['objects', 'mirror', 'reflections', 'floor']:
            all_meshes.extend(scene.get(key, []))
            
        if not all_meshes:
            raise ValueError("Scene contains no meshes")
        
        combined_mesh = join_meshes_as_scene(all_meshes)
        
        # 3. Rasterize to get Fragments
        # We don't need lights or materials, just geometry.
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1, 
        )
        
        rasterizer = MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        )
        
        fragments = rasterizer(combined_mesh)
        
        # 4. Extract Depth Map (for silhouettes)
        # zbuf shape: (N, H, W, K) -> take K=0
        depth_map = fragments.zbuf[..., 0].squeeze().cpu().numpy()
        
        # Mask out background (where pix_to_face is -1)
        pix_to_face = fragments.pix_to_face[..., 0].squeeze().cpu().numpy()
        background_mask = pix_to_face == -1
        
        # Normalize depth for edge detection (min-max normalization)
        depth_valid = depth_map[~background_mask]
        if depth_valid.size > 0:
            d_min, d_max = depth_valid.min(), depth_valid.max()
            if d_max > d_min:
                depth_norm = (depth_map - d_min) / (d_max - d_min)
                depth_norm = np.clip(depth_norm, 0, 1)
            else:
                depth_norm = np.zeros_like(depth_map)
        else:
            depth_norm = np.zeros_like(depth_map)
            
        depth_image = (depth_norm * 255).astype(np.uint8)
        depth_image[background_mask] = 0 # Set background to black

        # 5. Extract Normal Map (for internal creases)
        # Calculate vertex normals if not present
        if combined_mesh.verts_normals_list() is None or len(combined_mesh.verts_normals_list()) == 0:
             # This computes normals for the whole batch
             _ = combined_mesh.verts_normals_packed() 

        # Get normals per face vertex
        verts_normals = combined_mesh.verts_normals_packed() # (V, 3)
        faces = combined_mesh.faces_packed() # (F, 3)
        faces_normals = verts_normals[faces] # (F, 3, 3)
        
        # Interpolate normals to pixels
        # fragments.bary_coords: (N, H, W, K, 3)
        # fragments.pix_to_face: (N, H, W, K)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, 
            fragments.bary_coords, 
            faces_normals
        ) 
        # pixel_normals: (N, H, W, K, 3) -> (H, W, 3)
        normal_map = pixel_normals[0, ..., 0, :].squeeze().cpu().numpy()
        
        # Map normals [-1, 1] to [0, 255] for image processing
        normal_image = ((normal_map + 1) * 127.5).astype(np.uint8)
        normal_image[background_mask] = 0

        # 6. Compute Edges "Directly"
        # We find discontinuities in Depth (Silhouettes) and Normals (Creases)
        
        # Canny on Depth: Finds occlusion boundaries
        # Low thresholds because depth is clean
        edge_depth = cv2.Canny(depth_image, 50, 150) 
        
        # Canny on Normals: Finds surface changes
        # Normals are RGB-like, Canny works on intensity, so we process it as color or individual channels
        # A simple way is to take the max gradient across dimensions or run Canny on the visual representation
        edge_normal = cv2.Canny(normal_image, 100, 200)

        # Combine edges
        # You can use bitwise OR
        final_edges = cv2.bitwise_or(edge_depth, edge_normal)
        
        # Ensure lines are white, background black
        # (Canny already produces this: 255 for edge, 0 for background)

        # 7. Save
        image_path = self.output_dir / f"{output_prefix}.png"
        Image.fromarray(final_edges, mode="L").save(image_path)
        
        edge_map = EdgeMap()
        edge_map.image_path = str(image_path)
        
        return edge_map
