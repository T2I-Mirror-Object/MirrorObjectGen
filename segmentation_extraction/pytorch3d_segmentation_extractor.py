import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer.cameras import CamerasBase, FoVPerspectiveCameras
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer

from segmentation_extraction.segmentation_extractor import SegmentationExtractor, SegmentationMap


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


class PyTorch3DSegmentationExtractor(SegmentationExtractor):
    """
    Extract segmentation maps and metadata from PyTorch3D scenes.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        output_dir: str = "results/segmentation",
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
        R, T = look_at_view_transform(
            dist=self.camera_distance,        # distance from the origin
            elev=self.camera_elevation,       # in degrees
            azim=self.camera_azimuth,         # in degrees
        )

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

    def _extract_object_name(self, file_path: str) -> str:
        """
        Extract a clean object name from a file path.
        
        Args:
            file_path: Path to the object file (e.g., "data/dog_ahead.glb")
            
        Returns:
            Clean object name (e.g., "dog")
        """
        path = Path(file_path)
        # Get filename without extension
        name = path.stem
        # Remove common suffixes like "_ahead", "_front", etc.
        name = name.replace("_ahead", "").replace("_front", "").replace("_back", "")
        name = name.replace("_side", "").replace("_top", "").replace("_bottom", "")
        # Capitalize first letter
        return name.capitalize()

    def _get_segment_text(self, segment_type: str, index: int, total: int, object_name: Optional[str] = None) -> str:
        """
        Generate description text for a segment.

        Args:
            segment_type: Type of segment ('objects', 'mirror', 'reflections')
            index: Index within the segment type
            total: Total number of segments of this type
            object_name: Optional name of the object (for objects and reflections)

        Returns:
            Description string
        """
        if segment_type == "objects":
            if object_name:
                if total > 1:
                    return f"{object_name} standing in front of the mirror"
                return f"{object_name} standing in front of the mirror"
            else:
                if total > 1:
                    return f"Real object {index + 1} standing in front of the mirror."
                return "Real object standing in front of the mirror."
        elif segment_type == "mirror":
            return "The mirror frame surrounding the reflective surface."
        elif segment_type == "reflections":
            if object_name:
                if total > 1:
                    return f"The reflection of {object_name} visible inside the mirror"
                return f"The reflection of {object_name} visible inside the mirror"
            else:
                if total > 1:
                    return f"Reflection {index + 1} of the object visible inside the mirror."
                return "Reflection of the object visible inside the mirror."
        else:
            if object_name:
                return f"{object_name} (segment of type {segment_type}, index {index})"
            return f"Segment of type {segment_type}, index {index}."

    def extract_condition_map(self, scene: Dict[str, List[Meshes]],
                              output_prefix: str = "segmentation",
                              object_paths: Optional[List[str]] = None) -> ConditionMap:
        """
        Extract segmentation map and metadata from a scene.

        Args:
            scene: Dictionary with keys 'objects', 'mirror', 'reflections',
                   each containing a list of Meshes
            output_prefix: Prefix for output files
            object_paths: Optional list of file paths corresponding to objects in scene['objects'].
                         Used to extract object names for captions and segment descriptions.

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

        # Extract object names from paths if provided
        object_names = []
        if object_paths:
            object_names = [self._extract_object_name(path) for path in object_paths]
        else:
            object_names = [None] * len(scene.get('objects', []))

        # Prepare meshes and IDs
        # Order: objects, mirror, reflections
        all_meshes = []
        mesh_to_id = []
        segment_metadata = []  # Store (id, type, index_in_type, object_name) for later

        current_id = 1

        # Add objects
        objects_list = scene.get('objects', [])
        for i, mesh in enumerate(objects_list):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            obj_name = object_names[i] if i < len(object_names) else None
            segment_metadata.append((current_id, 'objects', i, len(objects_list), obj_name))
            current_id += 1

        # Add mirror
        for i, mesh in enumerate(scene.get('mirror', [])):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            segment_metadata.append((current_id, 'mirror', i, len(scene.get('mirror', [])), None))
            current_id += 1

        # Add reflections (match with corresponding objects)
        reflections_list = scene.get('reflections', [])
        for i, mesh in enumerate(reflections_list):
            all_meshes.append(mesh)
            mesh_to_id.append(current_id)
            # Reflection i corresponds to object i
            obj_name = object_names[i] if i < len(object_names) else None
            segment_metadata.append((current_id, 'reflections', i, len(reflections_list), obj_name))
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

        # Generate caption with object names
        if self.caption:
            caption = self.caption
        else:
            # Build caption from object names
            if object_names and any(name for name in object_names):
                valid_names = [name for name in object_names if name]
                if len(valid_names) == 1:
                    caption = f"A scene with a {valid_names[0].lower()} and its reflection in a mirror."
                elif len(valid_names) == 2:
                    caption = f"A scene with a {valid_names[0].lower()} and a {valid_names[1].lower()} and their reflections in a mirror."
                else:
                    names_str = ", ".join([name.lower() for name in valid_names[:-1]])
                    caption = f"A scene with {names_str}, and a {valid_names[-1].lower()} and their reflections in a mirror."
            else:
                caption = "A scene with objects and their mirror reflections."

        # Build JSON structure
        output_json = {
            "caption": caption,
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
            object_name = None

            for metadata in segment_metadata:
                if metadata[0] == inst_id:
                    segment_type = metadata[1]
                    segment_index = metadata[2]
                    segment_total = metadata[3]
                    object_name = metadata[4] if len(metadata) > 4 else None
                    break

            # Get description text
            text = self._get_segment_text(segment_type, segment_index, segment_total, object_name)

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