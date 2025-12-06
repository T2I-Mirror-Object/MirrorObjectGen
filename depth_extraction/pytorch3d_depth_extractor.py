import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer.cameras import CamerasBase, FoVPerspectiveCameras
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer

from depth_extraction.depth_extractor import DepthExtractor, DepthMap


class PyTorch3DDepthExtractor(DepthExtractor):
    """
    Extract depth maps from PyTorch3D scenes.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        output_dir: str = "results/depth",
        device: str = "cpu",
        camera_distance: float = 5.0,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        fov: float = 60.0,
        faces_per_pixel: int = 1,
        normalize: bool = True,
        invert: bool = True
    ):
        """
        Args:
            image_size: Output image size (H, W)
            output_dir: Directory to save depth maps
            device: Device for PyTorch ('cpu' or 'cuda')
            camera_distance: Distance of camera from origin
            camera_elevation: Camera elevation angle in degrees
            camera_azimuth: Camera azimuth angle in degrees
            fov: Field of view in degrees
            faces_per_pixel: Number of faces per pixel for rasterization
            normalize: Whether to normalize depth values to [0, 1]
            invert: Whether to invert depth (near=white, far=black)
        """
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.fov = fov
        self.faces_per_pixel = faces_per_pixel
        self.normalize = normalize
        self.invert = invert

    def _create_camera(self) -> CamerasBase:
        """Create a camera with specified parameters."""
        R, T = look_at_view_transform(
            dist=self.camera_distance,
            elev=self.camera_elevation,
            azim=self.camera_azimuth,
        )

        camera = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=self.fov
        )

        return camera

    def extract_depth_map(
        self,
        scene: Dict[str, List[Meshes]],
        output_prefix: str = "depth",
        object_paths: Optional[List[str]] = None,
        camera_params: Optional[Tuple[float, float, float]] = None,
        cameras: Optional[CamerasBase] = None
    ) -> DepthMap:
        """
        Extract depth map from a scene.

        Args:
            scene: Dictionary with keys 'objects', 'mirror', 'reflections',
                   each containing a list of Meshes
            output_prefix: Prefix for output files
            object_paths: Optional list of file paths (for compatibility, not used)
            camera_params: Optional tuple of (distance, elevation, azimuth) to override defaults
            cameras: Optional pre-configured CamerasBase object to use directly

        Returns:
            DepthMap with path to depth image
        """
        # Create camera
        if cameras is not None:
             camera = cameras
        elif camera_params is not None:
            dist, elev, azim = camera_params
            R, T = look_at_view_transform(dist, elev, azim)
            camera = FoVPerspectiveCameras(
                device=self.device, 
                R=R, 
                T=T, 
                fov=self.fov
            )
        else:
            camera = self._create_camera()

        # Create rasterizer with robust settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=0,  # Disable binning for robustness
            cull_backfaces=False
        )

        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ).to(self.device)

        # Prepare meshes - combine all scene elements
        all_meshes = []
        
        # Add objects
        for mesh in scene.get('objects', []):
            all_meshes.append(mesh)
        
        # Add mirror
        for mesh in scene.get('mirror', []):
            all_meshes.append(mesh)
        
        # Add reflections
        for mesh in scene.get('reflections', []):
            all_meshes.append(mesh)

        # Join all meshes into one scene
        if not all_meshes:
            raise ValueError("Scene contains no meshes")
        
        combined_mesh = join_meshes_as_scene(all_meshes)

        # Rasterize to get fragments (contains depth buffer)
        with torch.no_grad():
            fragments = rasterizer(combined_mesh)

        # Extract depth in NDC (Normalized Device Coordinates)
        # Shape: (N, H, W, K) where K is faces_per_pixel
        # Take the nearest layer (k=0)
        z_ndc = fragments.zbuf[..., 0]  # (N, H, W)
        pix_to_face = fragments.pix_to_face[..., 0]  # (N, H, W)
        valid = pix_to_face >= 0  # Mask of pixels that hit some face

        # Process depth for visualization
        depth_vis = z_ndc.clone()
        
        if valid.any():
            # Fill background with farthest valid depth
            far_val = depth_vis[valid].max()
            depth_vis[~valid] = far_val
            
            if self.normalize:
                # Normalize to [0, 1] range
                dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
                depth_img = (depth_vis - dmin) / (dmax - dmin + 1e-8)
                
                if self.invert:
                    # Invert so near=white, far=black
                    depth_img = 1.0 - depth_img
            else:
                depth_img = depth_vis
        else:
            # No valid pixels, create black image
            depth_img = torch.zeros_like(depth_vis)

        # Convert to numpy and scale to 0-255 for saving
        depth_np = depth_img[0].detach().cpu().numpy()  # Take first batch element
        depth_uint8 = (depth_np * 255).astype(np.uint8)

        # Save depth image as grayscale
        image_path = self.output_dir / f"{output_prefix}.png"
        Image.fromarray(depth_uint8, mode="L").save(image_path)

        # Create and return DepthMap
        depth_map = DepthMap()
        depth_map.image_path = str(image_path)

        return depth_map

    def extract_condition_map(
        self,
        scene: Dict[str, List[Meshes]],
        output_prefix: str = "depth",
        object_paths: Optional[List[str]] = None,
        camera_params: Optional[Tuple[float, float, float]] = None,
        cameras: Optional[CamerasBase] = None
    ) -> DepthMap:
        """
        Alias for extract_depth_map for compatibility with existing code.
        """
        return self.extract_depth_map(scene, output_prefix, object_paths, camera_params, cameras)
