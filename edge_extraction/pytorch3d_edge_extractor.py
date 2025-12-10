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
    MeshRenderer, 
    MeshRasterizer, 
    HardPhongShader,
    PointLights,
    Materials
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

from edge_extraction.edge_extractor import EdgeExtractor, EdgeMap


class PyTorch3DEdgeExtractor(EdgeExtractor):
    """
    Extract edge maps from PyTorch3D scenes using Canny edge detection on rendered RGB images.
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
        faces_per_pixel: int = 1,
        canny_low_threshold: int = 100,
        canny_high_threshold: int = 200,
        light_location: Tuple[float, float, float] = (0.0, 5.0, 5.0)
    ):
        """
        Args:
            image_size: Output image size (H, W)
            output_dir: Directory to save edge maps
            device: Device for PyTorch ('cpu' or 'cuda')
            camera_distance: Distance of camera from origin
            camera_elevation: Camera elevation angle in degrees
            camera_azimuth: Camera azimuth angle in degrees
            fov: Field of view in degrees
            faces_per_pixel: Number of faces per pixel for rasterization
            canny_low_threshold: Lower threshold for Canny edge detection
            canny_high_threshold: Upper threshold for Canny edge detection
            light_location: Location of the point light source
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
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.light_location = light_location

    def _create_camera(self) -> FoVPerspectiveCameras:
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

    def extract_edge_map(
        self,
        scene: Dict[str, List[Meshes]],
        output_prefix: str = "edge",
        object_paths: Optional[List[str]] = None,
        camera_params: Optional[Tuple[float, float, float]] = None,
        cameras: Optional[FoVPerspectiveCameras] = None
    ) -> EdgeMap:
        """
        Extract edge map from a scene.

        Args:
            scene: Dictionary with keys 'objects', 'mirror', 'reflections',
                   each containing a list of Meshes
            output_prefix: Prefix for output files
            object_paths: Optional list of file paths (not used essentially)
            camera_params: Optional tuple of (distance, elevation, azimuth) to override defaults
            cameras: Optional pre-configured CamerasBase object to use directly

        Returns:
            EdgeMap with path to edge image
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

        # Create renderer
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
        )

        lights = PointLights(device=self.device, location=[self.light_location])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=camera,
                lights=lights
            )
        )

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

        # Add floor - Optional: Floor often creates a horizon line which might be unwanted for object edges.
        # But if it's part of the scene, maybe we should keep it?
        # In depth map, floor is included. In segmentation, it seems not explicitly handled in the loop I saw?
        # Wait, depth_map_shap_e.py includes floor if present.
        for mesh in scene.get('floor', []):
            all_meshes.append(mesh)

        if not all_meshes:
            raise ValueError("Scene contains no meshes")
        
        combined_mesh = join_meshes_as_scene(all_meshes)
        
        # Assign a default white texture if none exists? 
        # Shap-E meshes usually have vertex colors. PyTorch3D uses them if textures are None.
        # We assume combined_mesh has valid textures or vertex colors.
        
        # Render RGB image
        images = renderer(combined_mesh) # (N, H, W, 4)
        
        # Take first image and connect to CPU
        image_rgb = images[0, ..., :3].detach().cpu().numpy() # (H, W, 3)
        image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # The background is usually black (0) in renderer output if no background color specified
        # But let's check. Default shader might set background. 
        # Actually MeshRenderer returns what shader returns. HardPhongShader blends with background?
        # Usually it returns RGBA. Alpha is 0 for background.
        
        # Let's use the alpha channel to ensure background is black if needed, 
        # but Canny works on intensity.
        # If alpha is present, we can composite against black.
        image_rgba = images[0].detach().cpu().numpy()
        alpha = image_rgba[..., 3:]
        image_rgb = image_rgba[..., :3] * alpha # premultiply alpha to ensure background is black
        image_rgb = (image_rgb * 255).astype(np.uint8)

        # Convert to Grayscale for Canny? Canny handles it, but typically expects grayscale.
        # But Canny on RGB also works (applied per channel? No, usually luminance).
        # We can pass RGB to cv2.Canny? No, cv2.Canny takes a single channel image.
        # So we convert to grayscale.
        
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny
        edges = cv2.Canny(image_gray, self.canny_low_threshold, self.canny_high_threshold)
        # edges is (H, W) with 0 and 255
        
        # Save edge image
        image_path = self.output_dir / f"{output_prefix}.png"
        Image.fromarray(edges, mode="L").save(image_path)
        
        # Create and return EdgeMap
        edge_map = EdgeMap()
        edge_map.image_path = str(image_path)
        
        return edge_map
