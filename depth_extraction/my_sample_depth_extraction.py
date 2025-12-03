import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.transforms import Transform3d, euler_angles_to_matrix

def reflect_points_across_plane(pts: torch.Tensor, n: torch.Tensor, p: torch.Tensor):
    """
    pts: (V,3) points in world coords
    n:   (3,)  plane normal (need not be unit; we normalize)
    p:   (3,)  a point on the plane
    Returns: (V,3) reflected points
    """
    n = n / (n.norm() + 1e-9)
    v = pts - p
    return pts - 2.0 * (v @ n)[:, None] * n[None, :]

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
obj_filename = "object.obj"
mirror_filename = "mirror_frame.obj"

# Load obj file
verts, faces_idx, _ = load_obj(obj_filename)
mirror_verts, mirror_faces_idx, _ = load_obj(mirror_filename)

R = euler_angles_to_matrix(torch.tensor([1.57, 0, 0], device=device), "XYZ")
obj_base_tf = Transform3d(device=device).rotate(R)
mirror_base_tf = Transform3d(device=device)

obj_verts_world_tmp    = obj_base_tf.transform_points(verts.to(device))          # (V,3)
mirror_verts_world_tmp = mirror_base_tf.transform_points(mirror_verts.to(device))# (V,3)

# 3) Compute AABB mins along Y for both meshes
obj_min_y    = obj_verts_world_tmp[:, 1].min()
mirror_min_y = mirror_verts_world_tmp[:, 1].min()

# 4) Translation needed to bottom-align (match mins on Y)
dy = (mirror_min_y - obj_min_y).item()

# 5) Final transforms (optionally, keep any Z offsets you already use)
obj_tf    = obj_base_tf.translate(0.0, dy, -2.0)         # move object up/down to align bottoms
mirror_tf = mirror_base_tf

faces = faces_idx.verts_idx
mirror_faces = mirror_faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

mirror_rgb = torch.ones_like(mirror_verts)[None]
mirror_textures = TexturesVertex(verts_features=mirror_rgb.to(device))


obj_mesh = Meshes(
    verts=[obj_tf.transform_points(verts.to(device))],
    faces=[faces.to(device)],
    textures=textures
)

mirror_mesh = Meshes(
    verts=[mirror_tf.transform_points(mirror_verts.to(device))],
    faces=[mirror_faces.to(device)],
    textures=mirror_textures
)

R, T = look_at_view_transform(5.0, 30, 150) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1,
    bin_size=0
)

lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

# --- Define your mirror plane ---
# Example: mirror plane is the XY plane at z = z0 (same Z as your frame plane)
z0 = -0.02  # e.g., where you placed the mirror; adjust to your scene
p  = torch.tensor([0.0, 0.0, z0], device=device)
n  = torch.tensor([0.0, 0.0, 1.0], device=device)  # outward normal

# Get the object vertices in world space (the already-transformed ones you render)
obj_verts_world = obj_mesh.verts_list()[0]  # (V,3) on the correct device/dtype
obj_faces       = obj_mesh.faces_list()[0]  # (F,3)

# Reflect vertices across the mirror plane
verts_reflected = reflect_points_across_plane(obj_verts_world, n, p)

# To keep the face winding correct after reflection (normals flip),
# swap two indices per face. This preserves “front face” orientation.
faces_reflected = obj_faces[:, [0, 2, 1]].contiguous()

# Optional: nudge the reflected copy a hair *behind* the mirror plane
# to avoid coplanar z-fighting with any glass quad you might add later.
epsilon = 1e-4
verts_reflected = verts_reflected - epsilon * n

# You can reuse the same textures; many people dim the reflection a bit:
reflected_color = torch.full_like(obj_verts_world, 0.8)  # 80% brightness
reflected_tex   = TexturesVertex(verts_features=reflected_color[None])

reflected_mesh = Meshes(
    verts=[verts_reflected],
    faces=[faces_reflected],
    textures=reflected_tex
)

# Compose the scene: frame, original object, and reflected duplicate
from pytorch3d.structures import join_meshes_as_scene
combined_mesh = join_meshes_as_scene([mirror_mesh, obj_mesh, reflected_mesh])

# Render as before
images = renderer(combined_mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].detach().cpu().numpy())
plt.axis("off")

# (Optional) make rasterization robust to heavy geometry to avoid bin overflow warnings
raster_settings = raster_settings.__class__(  # keep your existing values, tweak below as needed
    image_size=raster_settings.image_size,
    blur_radius=raster_settings.blur_radius,
    faces_per_pixel=raster_settings.faces_per_pixel,
    # choose ONE of these strategies:
    # bin_size=0,                         # safest: disable binning (slower)
    max_faces_per_bin=50000              # faster: raise capacity if bins overflow
)

rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

# 1) Rasterize to get fragments (contains zbuf, pix_to_face, etc.)
fragments = rasterizer(combined_mesh)

# 2) Depth in NDC (Near-to-Far). Shape: (N, H, W, K). Take nearest layer k=0.
z_ndc = fragments.zbuf[..., 0]                          # (N, H, W)
pix_to_face = fragments.pix_to_face[..., 0]             # (N, H, W)
valid = pix_to_face >= 0                                # mask of pixels that hit some face

# 3) Visualize NDC depth (nearer -> darker or lighter as you prefer)
depth_vis = z_ndc.clone()
# Fill background with farthest valid depth for nicer visualization
if valid.any():
    far_val = depth_vis[valid].max()
    depth_vis[~valid] = far_val
    # normalize to [0,1] for display; invert so near=white (optional)
    dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
    depth_img = (depth_vis - dmin) / (dmax - dmin + 1e-8)
    depth_img = 1.0 - depth_img
else:
    depth_img = torch.zeros_like(depth_vis)

plt.figure(figsize=(6,6))
plt.imshow(depth_img[0].detach().cpu().numpy(), cmap="gray")
plt.title("Depth (NDC, normalized)")
plt.axis("off")
plt.show()

# 4) (Optional) Convert NDC depth -> camera-space Z (metric in the camera frame)
# Build per-pixel NDC x,y grid in [-1, 1]
N, H, W = z_ndc.shape
ys = torch.linspace(1.0, -1.0, H, device=z_ndc.device)  # note flip to match image coords
xs = torch.linspace(-1.0, 1.0, W, device=z_ndc.device)
xv, yv = torch.meshgrid(xs, ys, indexing="xy")          # (W,H) -> we’ll permute

xy_ndc = torch.stack([xv, yv], dim=-1).permute(1,0,2)   # (H,W,2)
# stack x,y with z_ndc
xyz_ndc = torch.cat([xy_ndc[None, ...], z_ndc[..., None]], dim=-1)  # (N,H,W,3)

# Unproject to camera space; take Z
# world_coordinates=False => returns points in the camera coordinate system
cam_pts = cameras.unproject_points(
    xyz_ndc.reshape(N, H*W, 3),
    world_coordinates=False
).reshape(N, H, W, 3)
depth_camZ = cam_pts[..., 2]  # camera-space Z (forward depth)

# Visualize camera-space depth (valid-only normalization)
depth_camZ_vis = depth_camZ.clone()
if valid.any():
    # mask background to far val for display
    far_val = depth_camZ_vis[valid].max()
    depth_camZ_vis[~valid] = far_val
    cmin, cmax = depth_camZ_vis[valid].min(), depth_camZ_vis[valid].max()
    camZ_img = (depth_camZ_vis - cmin) / (cmax - cmin + 1e-8)
    camZ_img = 1.0 - camZ_img
else:
    camZ_img = torch.zeros_like(depth_camZ_vis)

plt.figure(figsize=(6,6))
plt.imshow(camZ_img[0].detach().cpu().numpy(), cmap="gray")
plt.title("Depth (camera-space Z, normalized)")
plt.axis("off")
plt.show()