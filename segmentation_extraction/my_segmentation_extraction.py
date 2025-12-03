from typing import Iterable, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer


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
        meshes: Sequence[Meshes],
        mesh_to_id: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
          meshes: list/sequence of Meshes (each can be batch=1 Meshes).
          mesh_to_id: optional iterable of ints (len == len(meshes)).
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
        faceid_to_id = torch.empty(total_faces + 1, dtype=torch.int32, device=scene.device if hasattr(scene, 'device') else None)
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
            # Convert boolean to indices via cumulative max trick
            # fallback: take k=0 then overwrite where no valid
            first_valid = torch.argmax(valid.to(torch.int64), dim=-1)  # (N,H,W), 0 if none valid or first valid at 0
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

# Assuming you already built: mirror_mesh, obj_mesh, reflected_mesh, cameras, device

inst_renderer = InstanceIDRenderer(
    cameras=cameras,
    image_size=(1024, 1024),
    faces_per_pixel=1,      # or >1 if you want multi-hit robustness
    bin_size=0,             # or set max_faces_per_bin=50000
    cull_backfaces=False
).to(device)

# Assign stable IDs for each mesh in order
# e.g., 1=mirror frame, 2=object, 3=reflection
ids = inst_renderer([mirror_mesh, obj_mesh, reflected_mesh],
                    mesh_to_id=[1, 2, 3])             # (N,H,W) int32
# If your camera batch is 1, take ids[0]
inst_id_img = ids[0]                                   # (H,W), int32 on device

# Save as an 8-bit visualization (purely for viewing):
import matplotlib.pyplot as plt
vis = inst_id_img.detach().cpu().numpy().astype('int32')   # for quick look

import numpy as np
import matplotlib as mpl

def id_color_table(max_id: int,
                   bg_rgb=(0, 0, 0),
                   cmap_name="tab20") -> np.ndarray:
    """
    Returns an array of shape (max_id+1, 3) with uint8 RGB colors.
      index 0 -> background color
      index k -> color for ID k
    """
    # Take the 20 discrete colors from tab20
    base = mpl.cm.get_cmap(cmap_name).colors  # length 20, float in [0,1]
    base = (np.array(base) * 255).astype(np.uint8)  # to 0-255 uint8

    table = np.zeros((max_id + 1, 3), dtype=np.uint8)
    table[0] = np.array(bg_rgb, dtype=np.uint8)      # background
    for k in range(1, max_id + 1):
        table[k] = base[(k - 1) % len(base)]         # wrap if >20 IDs
    return table

# Example: build mapping and list colors actually used in your image
vis = inst_id_img.detach().cpu().numpy().astype(np.int32)  # (H,W)
palette = id_color_table(max_id=int(vis.max()), bg_rgb=(0,0,0))

# Dictionary: {id: (R,G,B)}
id_to_rgb = {int(k): tuple(map(int, palette[k])) for k in np.unique(vis)}

from PIL import Image
import json
import numpy as np

# --- Inputs assumed already defined upstream ---
# vis:            (H, W) int32 instance ID map (0 = background)
# palette:        np.ndarray of shape (max_id+1, 3), dtype=uint8 (palette[k] = RGB for ID k)
# id_to_rgb:      {id: (R, G, B)} mapping derived from palette, e.g. {k: tuple(palette[k]) for k in np.unique(vis)}
# NOTE: These should be built from the SAME palette you used to colorize the mask.

# 1) Save a strict RGB segmentation image (no Matplotlib colormap)
rgb = palette[vis]  # (H, W, 3) uint8
Image.fromarray(rgb, mode="RGB").save("instance_ids_rgb.png")

# 2) Define human-readable text for each instance ID (customize as needed)
#    Provide texts only for the NON-BACKGROUND ids you expect; others will get a generic fallback.
segment_texts = {
    1: "The mirror frame surrounding the reflective surface.",
    2: "The real dog standing in front of the mirror.",
    3: "The dog's reflection visible inside the mirror."
}

# 3) Build the targeted JSON structure
output_json = {
    "caption": "A photo of a dog and its reflection in the mirror. The dog stands close to the mirror in soft indoor light, and its reflection appears clearly on the glass within a simple frame. The background is gently out of focus, drawing attention to the dog and the mirrored image.",
    "seed": 53,
    "segments_info": []
}

# Sort unique IDs and skip background (0)
unique_ids = sorted(int(k) for k in np.unique(vis) if int(k) != 0)

for inst_id in unique_ids:
    # Use the EXACT RGB used to render the RGB mask
    if inst_id in id_to_rgb:
        color_rgb = list(map(int, id_to_rgb[inst_id]))
    else:
        # Fallback to palette table if not present in dict
        color_rgb = list(map(int, palette[inst_id].tolist()))

    text = segment_texts.get(inst_id, f"Object with instance ID {inst_id}.")
    output_json["segments_info"].append({
        "color": color_rgb,   # [R, G, B]
        "text": text
    })

print(output_json)
# 4) Save the targeted JSON
with open("instance_id_colors.json", "w") as f:
    json.dump(output_json, f, indent=2)

print("Saved instance_ids_rgb.png and instance_id_colors.json")

# 5) (Optional) Sanity check: ensure that for each ID, at least one pixel matches the palette color in the saved RGB
#    This helps catch channel-order or palette mismatches early.
for inst_id in unique_ids:
    # find one pixel of this ID
    ys, xs = np.where(vis == inst_id)
    if ys.size > 0:
        y0, x0 = int(ys[0]), int(xs[0])
        assert (rgb[y0, x0] == np.array(palette[inst_id], dtype=np.uint8)).all(), \
            f"Color mismatch for ID {inst_id}: rgb[{y0},{x0}] != palette[{inst_id}]"

