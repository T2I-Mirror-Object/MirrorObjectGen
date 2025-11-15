import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase, FoVPerspectiveCameras
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer

from condition_extraction.condition_extractor import ConditionExtractor, ConditionMap


class InstanceIDRenderer(nn.Module):
    """
    Render per-pixel instance IDs for a list of meshes composed into one scene.

    IDs:
      0 -> background
      mesh_to_id[i] (or i+1 if None) -> pixels covered by meshes[i]
    """
    def __init__(
        self,
        cameras: CamerasBase,
        image_size: Tuple[int, int],
        *,
        faces_per_pixel: int = 1,
        blur_radius: float = 1e-7,
        max_faces_per_bin: Optional[int] = None,
        bin_size: Optional[int] = None,
        cull_backfaces: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=max_faces_per_bin,
            bin_size=bin_size,
            cull_backfaces=cull_backfaces,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)

    def to(self, device):
        # Move buffers to device (Cameras in the rasterizer hold device state)
        self.rasterizer = self.rasterizer.to(device)
        return super().to(device)

    @torch.no_grad()
    def forward(
        self,
        meshes: List[Meshes],
        mesh_to_id: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
          meshes: list/sequence of Meshes (each can be batch=1 Meshes).
          mesh_to_id: optional list of ints (len == len(meshes)).
                      If None, uses 1..len(meshes).

        Returns:
          ids: (N, H, W) int32 per-pixel instance IDs. N is camera batch size.
        """
        assert isinstance(meshes, (list, tuple)) and len(meshes) > 0, "Pass a non-empty list of Meshes."

        # Join into one scene (faces become concatenated in order).
        scene = join_meshes_as_scene(meshes)

        # Build mapping: face_index (packed) -> instance ID
        # face indices of the joined mesh are [0 .. total_faces-1].
        faces_per_mesh = [m.num_faces_per_mesh().sum().item() for m in meshes]
        total_faces = sum(faces_per_mesh)

        if mesh_to_id is None:
            mesh_to_id = list(range(1, len(meshes) + 1))
        else:
            mesh_to_id = list(mesh_to_id)
            assert len(mesh_to_id) == len(meshes), "mesh_to_id length must match meshes."

        # Create mapping vector with a leading 0 for background (so we can index +1 safely).
        # faceid_to_id[k+1] gives the instance ID of packed face k; faceid_to_id[0] = 0 (bg).
        device = scene.device if hasattr(scene, 'device') else torch.device('cpu')
        faceid_to_id = torch.empty(total_faces + 1, dtype=torch.int32, device=device)
        faceid_to_id[0] = 0
        cursor = 1
        for fcnt, inst_id in zip(faces_per_mesh, mesh_to_id):
            faceid_to_id[cursor:cursor + fcnt] = int(inst_id)
            cursor += fcnt

        # Rasterize
        fragments = self.rasterizer(scene, **kwargs)  # pix_to_face: (N,H,W,K), -1 for bg
        pix_to_face = fragments.pix_to_face  # (N,H,W,K), int64

        # Choose the closest face per pixel.
        # If faces_per_pixel>1, take the first valid (k where pix_to_face>=0), else k=0.
        if pix_to_face.shape[-1] == 1:
            hit_face = pix_to_face[..., 0]  # (N,H,W)
        else:
            # mask valid faces
            valid = pix_to_face >= 0  # (N,H,W,K)
            # index of first valid along K; if none valid -> keep -1
            first_valid = torch.argmax(valid.to(torch.int64), dim=-1)  # (N,H,W)
            # gather faces at first_valid
            hit_face = torch.gather(
                pix_to_face,
                dim=-1,
                index=first_valid.unsqueeze(-1)
            ).squeeze(-1)  # (N,H,W)
            # where no valid at all, set -1 explicitly
            none_valid = ~valid.any(dim=-1)
            hit_face[none_valid] = -1

        # Map faces -> instance IDs (shift by +1 to use bg=0 slot)
        # hit_face == -1 -> index 0 (bg)
        ids = faceid_to_id[(hit_face + 1).clamp(min=0)]  # (N,H,W), int32

        return ids


class PyTorch3DConditionExtractor(ConditionExtractor):
    """
    Extract segmentation maps and metadata from PyTorch3D scenes.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        output_dir: str = "results",
        device: str = "cpu",
        camera_distance: float = 5.0,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        fov: float = 60.0,
        faces_per_pixel: int = 1,
        segment_descriptions: Optional[Dict[str, str]] = None,
        caption: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            image_size: Output image size (H, W)
            output_dir: Directory to save segmentation maps and JSON files
            device: Device for PyTorch ('cpu' or 'cuda')
            camera_distance: Distance of camera from origin
            camera_elevation: Camera elevation angle in degrees
            camera_azimuth: Camera azimuth angle in degrees
            fov: Field of view in degrees
            faces_per_pixel: Number of faces per pixel for rasterization
            segment_descriptions: Dict mapping segment types to descriptions
                                  e.g., {'objects': 'description', 'mirror': 'description', 'reflections': 'description'}
            caption: Overall scene caption for JSON metadata
            seed: Random seed for reproducibility
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
        self.segment_descriptions = segment_descriptions or {}
        self.caption = caption
        self.seed = seed

    def _create_camera(self) -> CamerasBase:
        """Create a camera with specified parameters."""
        # Convert elevation and azimuth to radians
        elev_rad = np.deg2rad(self.camera_elevation)
        azim_rad = np.deg2rad(self.camera_azimuth)

        # Calculate camera position
        x = self.camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = self.camera_distance * np.sin(elev_rad)
        z = self.camera_distance * np.cos(elev_rad) * np.cos(azim_rad)

        R = torch.eye(3).unsqueeze(0)  # Identity rotation (we'll use position)
        T = torch.tensor([[x, y, z]], dtype=torch.float32)

        camera = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=self.fov
        )

        return camera

    def _create_color_palette(self, max_id: int, bg_rgb=(0, 0, 0), cmap_name="tab20") -> np.ndarray:
        """
        Create a color palette for instance IDs.

        Args:
            max_id: Maximum instance ID
            bg_rgb: Background color RGB tuple
            cmap_name: Matplotlib colormap name

        Returns:
            Array of shape (max_id+1, 3) with uint8 RGB colors
        """
        import matplotlib as mpl

        # Get colormap
        base = mpl.cm.get_cmap(cmap_name).colors  # length 20, float in [0,1]
        base = (np.array(base) * 255).astype(np.uint8)  # to 0-255 uint8

        table = np.zeros((max_id + 1, 3), dtype=np.uint8)
        table[0] = np.array(bg_rgb, dtype=np.uint8)  # background
        for k in range(1, max_id + 1):
            table[k] = base[(k - 1) % len(base)]  # wrap if >20 IDs
        return table

    def _get_segment_text(self, segment_type: str, index: int, total: int) -> str:
        """
        Generate description text for a segment.

        Args:
            segment_type: Type of segment ('objects', 'mirror', 'reflections')
            index: Index within the segment type
            total: Total number of segments of this type

        Returns:
            Description string
        """
        if segment_type in self.segment_descriptions:
            base_desc = self.segment_descriptions[segment_type]
            if total > 1:
                return f"{base_desc} (item {index + 1} of {total})"
            return base_desc
        else:
            # Default descriptions
            if segment_type == "objects":
                if total > 1:
                    return f"Real object {index + 1} standing in front of the mirror."
                return "Real object standing in front of the mirror."
            elif segment_type == "mirror":
                return "The mirror frame surrounding the reflective surface."
            elif segment_type == "reflections":
                if total > 1:
                    return f"Reflection {index + 1} of the object visible inside the mirror."
                return "Reflection of the object visible inside the mirror."
            else:
                return f"Segment of type {segment_type}, index {index}."

    def extract_condition_map(self, scene: Dict[str, List[Meshes]],
                              output_prefix: str = "segmentation") -> ConditionMap:
        """
        Extract segmentation map and metadata from a scene.

        Args:
            scene: Dictionary with keys 'objects', 'mirror', 'reflections',
                   each containing a list of Meshes
            output_prefix: Prefix for output files

        Returns:
            ConditionMap with paths to image and JSON files
        """
        # Create camera
        camera = self._create_camera()

        # Create renderer
        renderer = InstanceIDRenderer(
            cameras=camera,
            image_size=self.image_size,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=0,
            cull_backfaces=False
        ).to(self.device)

        # Prepare meshes and IDs
        # Order: objects, mirror, reflections
        all_meshes = []
        mesh_to_id = []
        segment_metadata = []  # Store (id, type, index_in_type) for later

        current_id = 1

        # Add objects
        for i, mesh in enumerate(scene.get('objects', [])):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            segment_metadata.append((current_id, 'objects', i, len(scene.get('objects', []))))
            current_id += 1

        # Add mirror
        for i, mesh in enumerate(scene.get('mirror', [])):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            segment_metadata.append((current_id, 'mirror', i, len(scene.get('mirror', []))))
            current_id += 1

        # Add reflections
        for i, mesh in enumerate(scene.get('reflections', [])):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            segment_metadata.append((current_id, 'reflections', i, len(scene.get('reflections', []))))
            current_id += 1

        # Render instance IDs
        ids = renderer(all_meshes, mesh_to_id=mesh_to_id)  # (N, H, W)
        inst_id_img = ids[0]  # (H, W) - take first camera batch

        # Convert to numpy
        vis = inst_id_img.detach().cpu().numpy().astype(np.int32)

        # Create color palette
        max_id = int(vis.max())
        palette = self._create_color_palette(max_id, bg_rgb=(0, 0, 0))

        # Create RGB segmentation image
        rgb = palette[vis]  # (H, W, 3) uint8

        # Save RGB image
        image_path = self.output_dir / f"{output_prefix}.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)

        # Build id_to_rgb mapping
        id_to_rgb = {int(k): tuple(map(int, palette[k])) for k in np.unique(vis)}

        # Build JSON structure
        output_json = {
            "caption": self.caption or "A scene with objects and their mirror reflections.",
            "segments_info": []
        }

        if self.seed is not None:
            output_json["seed"] = self.seed

        # Get unique IDs (skip background 0)
        unique_ids = sorted(int(k) for k in np.unique(vis) if int(k) != 0)

        for inst_id in unique_ids:
            # Get RGB color
            if inst_id in id_to_rgb:
                color_rgb = list(map(int, id_to_rgb[inst_id]))
            else:
                color_rgb = list(map(int, palette[inst_id].tolist()))

            # Find metadata for this ID
            segment_type = "unknown"
            segment_index = 0
            segment_total = 1

            for metadata in segment_metadata:
                if metadata[0] == inst_id:
                    segment_type = metadata[1]
                    segment_index = metadata[2]
                    segment_total = metadata[3]
                    break

            # Get description text
            text = self._get_segment_text(segment_type, segment_index, segment_total)

            output_json["segments_info"].append({
                "color": color_rgb,
                "text": text
            })

        # Save JSON
        json_path = self.output_dir / f"{output_prefix}.json"
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)

        # Create and return ConditionMap
        condition_map = ConditionMap()
        condition_map.image_path = str(image_path)
        condition_map.json_path = str(json_path)

        return condition_map


if __name__ == "__main__":
    # Example usage
    from scene_composition.pytorch3d_scene_composition import SceneComposition

    # Create scene
    scene_compositor = SceneComposition(device="cpu")
    scene = scene_compositor.compose_scene([
        "data/dog_ahead.glb",
        "data/cat_ahead.glb",
        "data/lamp.glb"
    ])

    # Extract condition map
    extractor = PyTorch3DConditionExtractor(
        image_size=(1024, 1024),
        output_dir="results",
        device="cpu",
        camera_distance=10.0,
        camera_elevation=15.0,
        camera_azimuth=0.0,
        segment_descriptions={
            'objects': 'Real object in front of the mirror',
            'mirror': 'The mirror frame',
            'reflections': 'Reflection visible in the mirror'
        },
        caption="A scene with multiple objects and their reflections in a mirror.",
        seed=42
    )

    condition_map = extractor.extract_condition_map(scene, output_prefix="scene_segmentation")

    print(f"Segmentation saved to: {condition_map.image_path}")
    print(f"Metadata saved to: {condition_map.json_path}")
