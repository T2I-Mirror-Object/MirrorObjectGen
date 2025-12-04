import torch
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_parser.text_parser_impl_2 import TextParserImpl2
from text_to_3d.lgm_full import LGMFull
from scene_composition.pytorch3d_scene_composition import SceneComposition

from pytorch3d.structures import Pointclouds, Meshes, join_pointclouds_as_scene
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer
)

class PointCloudSceneComposition(SceneComposition):
    """
    Adapted SceneComposition for Pointclouds.
    Inherits from SceneComposition to reuse constants and some logic,
    but overrides methods to handle Pointclouds.
    """
    def _load_objects(self, object_paths: List[str]) -> List[Pointclouds]:
        """Load point clouds from file paths."""
        pcls = []
        io = IO()
        
        for object_path in object_paths:
            # Load as Pointcloud
            # LGM outputs PLY which pytorch3d can load
            try:
                pcl = io.load_pointcloud(object_path, device=self.device)
                pcls.append(pcl)
            except Exception as e:
                print(f"Error loading {object_path}: {e}")
                # Fallback: try loading as mesh and sampling if it happens to be a mesh
                try:
                    mesh = io.load_mesh(object_path, device=self.device)
                    pcl = sample_points_from_meshes(mesh, num_samples=10000)
                    pcls.append(pcl)
                except:
                    raise e
        return pcls

    def _get_mesh_bounds(self, pcl: Pointclouds) -> torch.Tensor:
        """Get bounding box of a point cloud."""
        # Pointclouds.get_bounding_boxes() returns (N, 3, 2) -> (min, max)
        # We want [min, max] which is (2, 3)
        bbox = pcl.get_bounding_boxes()[0].t() # (3, 2) -> (2, 3) if we transpose? 
        # Wait, get_bounding_boxes returns (N, 3, 2) where last dim is (min, max).
        # So bbox[0] is (3, 2). Transpose gives (2, 3) -> [min_vec, max_vec]
        return bbox.t()

    def _get_mesh_centroid(self, pcl: Pointclouds) -> torch.Tensor:
        """Get centroid of a point cloud."""
        return pcl.points_packed().mean(dim=0)

    def _apply_translation(self, pcl: Pointclouds, translation: torch.Tensor) -> Pointclouds:
        """Apply translation to a point cloud."""
        points = pcl.points_packed()
        new_points = points + translation
        return Pointclouds(points=[new_points], features=pcl.features_list())

    def _apply_transform(self, pcl: Pointclouds, transform_matrix: torch.Tensor) -> Pointclouds:
        """Apply 4x4 transformation matrix."""
        # Not strictly needed if we only use translation and rotation separately, 
        # but good to have.
        points = pcl.points_packed()
        ones = torch.ones((points.shape[0], 1), device=self.device)
        points_homo = torch.cat([points, ones], dim=1)
        new_points_homo = torch.matmul(points_homo, transform_matrix.t())
        new_points = new_points_homo[:, :3]
        return Pointclouds(points=[new_points], features=pcl.features_list())

    def _put_single_object_on_plane_xz(self, pcl: Pointclouds) -> Pointclouds:
        """Place object on XZ plane."""
        bounds = self._get_mesh_bounds(pcl)
        min_y = bounds[0, 1]
        translation = torch.tensor([0.0, -min_y, 0.0], device=self.device)
        return self._apply_translation(pcl, translation)
    
    # _put_objects_on_plane_xz, _shift_objects_to_center, _put_objects_next_to_each_other
    # reuse the logic but call the overridden methods.
    # Since the parent class uses type hints for Meshes, we might need to copy-paste 
    # if we want to be strict, but Python is dynamic.
    # However, the parent methods call self._get_mesh_bounds etc, which we overrode.
    # So we can reuse them if we just pass pcls instead of meshes.
    
    def _add_random_rotation(self, pcls: List[Pointclouds]) -> List[Pointclouds]:
        """Add random rotation around Y-axis."""
        rotated_pcls = []
        for pcl in pcls:
            angle = np.random.uniform(self.min_angle, self.max_angle)
            centroid = self._get_mesh_centroid(pcl)
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=torch.float32, device=self.device)
            
            points = pcl.points_packed()
            centered = points - centroid
            rotated = torch.matmul(centered, R.t())
            final = rotated + centroid
            
            rotated_pcls.append(Pointclouds(points=[final], features=pcl.features_list()))
        return rotated_pcls

    def _create_mirror_frame(self, pcls: List[Pointclouds]) -> Pointclouds:
        """Create mirror frame as point cloud."""
        # Calculate dimensions using parent logic (which calls our overridden bounds)
        total_width = self._total_objects_width(pcls) + self.mirror_gap_side * 2
        total_height = self._max_objects_height(pcls) + self.mirror_gap_top
        
        # Create mesh first using parent method
        frame_mesh = self._make_mirror_frame(total_width, total_height, self.mirror_thickness)
        
        # Sample points from mesh to get point cloud
        # Use enough points to make it look solid in depth map
        frame_pcl = sample_points_from_meshes(frame_mesh, num_samples=50000)
        
        # Move mirror behind objects
        translation = torch.tensor([0, 0, -self.mirror_gap_ahead], dtype=torch.float32, device=self.device)
        frame_pcl = self._apply_translation(frame_pcl, translation)
        
        return frame_pcl

    def _calculate_objects_reflection(self, pcls: List[Pointclouds]) -> List[Pointclouds]:
        """Create mirror reflections."""
        reflected_pcls = []
        for pcl in pcls:
            points = pcl.points_packed().clone()
            features = pcl.features_list()
            if features:
                features = [f.clone() for f in features]
            
            # Reflect along Z (negate Z)
            reflection_matrix = torch.diag(torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device))
            reflected_points = torch.matmul(points, reflection_matrix.t())
            
            # Move behind mirror
            translation = torch.tensor([0, 0, -2 * self.mirror_gap_ahead], dtype=torch.float32, device=self.device)
            reflected_points = reflected_points + translation
            
            reflected_pcls.append(Pointclouds(points=[reflected_points], features=features))
            
        return reflected_pcls

    def compose_scene(self, object_paths: List[str]) -> Dict[str, List[Pointclouds]]:
        # Same logic as parent, but using our methods
        objects = self._load_objects(object_paths)
        objects = self._put_objects_on_plane_xz(objects)
        objects = self._add_random_rotation(objects)
        objects = self._put_objects_next_to_each_other(objects)
        
        mirror_frame = self._create_mirror_frame(objects)
        reflections = self._calculate_objects_reflection(objects)
        
        return {
            'objects': objects,
            'mirror': [mirror_frame],
            'reflections': reflections
        }


class PointCloudDepthExtractor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        output_dir: str = "results/depth",
        device: str = "cpu",
        camera_distance: float = 5.0,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        fov: float = 60.0,
        normalize: bool = True,
        invert: bool = True
    ):
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.fov = fov
        self.normalize = normalize
        self.invert = invert

    def extract_depth_map(
        self,
        scene: Dict[str, List[Pointclouds]],
        output_prefix: str = "depth"
    ) -> str:
        # Create camera
        R, T = look_at_view_transform(
            dist=self.camera_distance,
            elev=self.camera_elevation,
            azim=self.camera_azimuth,
        )
        camera = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)

        # Rasterizer settings
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=0.01, # Adjust point radius
            points_per_pixel=10
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)

        # Combine all point clouds
        all_pcls = []
        for key in ['objects', 'mirror', 'reflections']:
            all_pcls.extend(scene.get(key, []))
        
        if not all_pcls:
            raise ValueError("Scene is empty")
            
        combined_pcl = join_pointclouds_as_scene(all_pcls)

        # Rasterize
        with torch.no_grad():
            fragments = rasterizer(combined_pcl)

        # Extract depth
        # fragments.zbuf: (N, H, W, K)
        z_buf = fragments.zbuf[..., 0] # Closest point
        idx = fragments.idx[..., 0] # Index of closest point
        
        valid = idx >= 0
        
        depth_vis = z_buf.clone()
        
        if valid.any():
            far_val = depth_vis[valid].max()
            depth_vis[~valid] = far_val
            
            if self.normalize:
                dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
                depth_img = (depth_vis - dmin) / (dmax - dmin + 1e-8)
                if self.invert:
                    depth_img = 1.0 - depth_img
            else:
                depth_img = depth_vis
        else:
            depth_img = torch.zeros_like(depth_vis)

        # Save
        depth_np = depth_img[0].detach().cpu().numpy()
        depth_uint8 = (depth_np * 255).astype(np.uint8)
        
        image_path = self.output_dir / f"{output_prefix}.png"
        Image.fromarray(depth_uint8, mode="L").save(image_path)
        
        return str(image_path)


def generate_depth_for_prompt(
    prompt,
    output_dir="results",
    camera_distance=5.0,
    camera_elevation=25.0,
    camera_azimuth=10.0
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing prompt: {prompt}")
    
    # Parse
    text_parser = TextParserImpl2()
    obj_name_list = text_parser.parse(prompt)
    
    # Generate 3D (LGM)
    lgm = LGMFull(device="cuda" if torch.cuda.is_available() else "cpu")
    obj_paths = lgm.convert_multiple_texts_to_3d(
        texts=obj_name_list,
        output_dir=f"{output_dir}/lgm"
    )
    
    print("Objects created by LGM.")
    
    # Compose Scene (Point Clouds)
    compositor = PointCloudSceneComposition(
        gap=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    scene = compositor.compose_scene(obj_paths)
    
    # Extract Depth
    extractor = PointCloudDepthExtractor(
        image_size=(1024, 1024),
        output_dir=f"{output_dir}/depth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth
    )
    
    depth_path = extractor.extract_depth_map(scene, output_prefix="scene_depth_lgm")
    print(f"Depth map saved to: {depth_path}")
    return depth_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="a teddy bear in front of the mirror")
    parser.add_argument('--output-dir', type=str, default="results")
    parser.add_argument('--camera-distance', type=float, default=3.0)
    parser.add_argument('--camera-elevation', type=float, default=25.0)
    parser.add_argument('--camera-azimuth', type=float, default=10.0)
    
    args = parser.parse_args()
    
    generate_depth_for_prompt(
        args.prompt,
        args.output_dir,
        args.camera_distance,
        args.camera_elevation,
        args.camera_azimuth
    )
