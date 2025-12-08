import torch
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import pipeline

def image_to_point_cloud(image_path, output_pcd_path="object.ply"):
    """
    1. Estimates depth from a single image.
    2. Back-projects pixels to 3D points.
    3. Saves as a Point Cloud.
    """
    print(f"--- Processing {image_path} ---")
    
    # 1. LOAD IMAGE & DEPTH MODEL
    # We use the 'Small' version for speed, use 'Large' for best shape
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cuda")
    pil_image = Image.open(image_path).convert("RGB")
    
    # Run Inference
    depth_map = pipe(pil_image)["depth"]
    
    # 2. PREPARE DATA FOR OPEN3D
    width, height = pil_image.size
    
    # Convert to Open3D Image formats
    # Depth Anything outputs relative depth (0-255 grayscale usually). 
    # We invert it because usually brighter = closer in these maps, 
    # but Open3D expects 'depth' (distance from camera).
    # NOTE: Check your model output. Depth Anything usually gives 'inverse depth' (disparity).
    depth_np = np.array(depth_map)
    
    # Invert/Scale depth: 
    # Real depth = 1 / predicted_depth (roughly, for disparity models)
    # We simply normalize it to a reasonable metric range (e.g., 1.0m to 5.0m)
    depth_np = depth_np.astype(np.float32)
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) # 0 to 1
    depth_np = 1.0 + (depth_np * 4.0) # Map to 1.0m -> 5.0m range
    
    # Create Open3D Images
    o3d_color = o3d.geometry.Image(np.array(pil_image))
    o3d_depth = o3d.geometry.Image(depth_np)
    
    # 3. CREATE RGBD IMAGE
    # convert_rgb_to_intensity=False ensures we keep color
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, 
        depth_scale=1.0, 
        depth_trunc=10.0, 
        convert_rgb_to_intensity=False
    )
    
    # 4. DEFINE CAMERA INTRINSICS
    # We approximate a standard camera (FOV ~60 deg)
    fx = fy = max(width, height)  
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 5. BACK-PROJECT TO POINT CLOUD
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )
    
    # Rotate it to be upright (Open3D standard view is different from image view)
    # Usually we flip Y to point up
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 6. SAVE
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Saved Point Cloud to {output_pcd_path}")
    
    return pcd


def point_cloud_to_mesh(pcd, output_mesh_path="reconstructed_mesh.obj"):
    """
    1. Post-processes the point cloud (outlier removal).
    2. Estimates normals (Crucial!).
    3. Runs Poisson Surface Reconstruction.
    4. Saves as .obj
    """
    print(f"--- Meshing Point Cloud ---")

    # 1. OUTLIER REMOVAL (Clean up flying pixels)
    # Remove points that are far from their neighbors
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd.select_by_index(ind)
    
    # 2. ESTIMATE NORMALS (Required for Meshing)
    # The algorithm needs to know which way is "out"
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    # Orient normals towards the camera so the mesh isn't inside-out
    pcd_clean.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # 3. POISSON RECONSTRUCTION
    # depth=8 or 9 is good detail. Higher = more detail but longer time.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=8
    )
    
    # 4. CLEANUP
    # Poisson creates a 'bubble' that might include extra space. 
    # We remove vertices with low density (where there were no original points)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 5. SAVE
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"Saved Mesh to {output_mesh_path}")
    
    return mesh

# --- USAGE ---
# 1. Generate Cloud
pcd = image_to_point_cloud("my_chair_image.jpg")

# 2. Convert to Mesh
mesh = point_cloud_to_mesh(pcd)

# 3. (Optional) Visualize
o3d.visualization.draw_geometries([mesh])