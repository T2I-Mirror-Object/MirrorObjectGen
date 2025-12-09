import torch
import torch.nn as nn
import yaml
import math
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader,
    BlendParams,
    look_at_view_transform
)
from pytorch3d.structures import join_meshes_as_scene

# --- Helper to initialize raw parameters for Sigmoid ---
def inverse_sigmoid(x):
    # Numerical stability clamp
    x = min(max(x, 1e-4), 1 - 1e-4)
    return math.log(x / (1 - x))

def load_config(config_path="configs/camera-optimization.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

class RobustViewOptimizer(nn.Module):
    """
    ### IMPROVEMENT: Robust Optimizer
    Uses Sigmoid activation to strictly constrain Elevation and Distance.
    This prevents the camera from flipping upside down or clipping through the object.
    """
    def __init__(self, device="cuda", config=None):
        super().__init__()
        self.device = device
        
        # Defaults
        init_params = config.get("initial_parameters", {}) if config else {}
        constraints = config.get("constraints", {}) if config else {}
        
        # Targets/Start values
        start_dist = init_params.get("distance", 3.5)
        start_elev = init_params.get("elevation", 20.0)
        start_azim = init_params.get("azimuth", 15.0)
        if -10 < start_azim < 10:
            print(f"Adjusting start azimuth from {start_azim} to 15.0 to avoid blind spot.")
            start_azim = 15.0

        # 1. Define Hard Constraints (Min/Max)
        self.min_dist = 1.0
        self.max_dist = constraints.get("target_distance", 3.5) * 2.0 # Allow some range
        
        self.min_elev = 5.0  # Don't go below 5 degrees (floor clipping)
        self.max_elev = constraints.get("max_elevation", 85.0) # Don't go exactly overhead (90)

        # 2. Initialize Raw Parameters using Inverse Sigmoid
        # This ensures the optimization starts EXACTLY at start_elev/start_dist
        
        # Normalize start values to 0-1 range for initialization
        norm_dist = (start_dist - self.min_dist) / (self.max_dist - self.min_dist)
        norm_elev = (start_elev - self.min_elev) / (self.max_elev - self.min_elev)
        
        self.raw_dist = nn.Parameter(torch.tensor(inverse_sigmoid(norm_dist), device=device))
        self.raw_elev = nn.Parameter(torch.tensor(inverse_sigmoid(norm_elev), device=device))
        self.azim = nn.Parameter(torch.tensor(float(start_azim), device=device))
        
    def forward(self):
        # 3. Apply Constraints (Sigmoid -> Scale -> Shift)
        # The output is mathematically guaranteed to stay within [min, max]
        actual_dist = torch.sigmoid(self.raw_dist) * (self.max_dist - self.min_dist) + self.min_dist
        actual_elev = torch.sigmoid(self.raw_elev) * (self.max_elev - self.min_elev) + self.min_elev
        
        # Convert to R, T
        R, T = look_at_view_transform(actual_dist, actual_elev, self.azim)
        
        # Update camera
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # Return the actual values too so we can log them
        return cameras, actual_dist, actual_elev

def optimize_view(scene, device='cuda', config_path="configs/camera-optimization.yaml"):
    # Load configuration
    config = load_config(config_path)

    # Extract meshes
    reflection_meshes = scene.get('reflections', [])
    if not reflection_meshes:
        print("Warning: No reflection meshes found. Using default.")
        return 3.5, 20.0, 0.0
        
    reflected_chair_mesh = join_meshes_as_scene(reflection_meshes).to(device)
    
    # Initialize Robust Model
    optimizer_model = RobustViewOptimizer(device=device, config=config).to(device)
    
    # Optimization Params
    opt_config = config.get("optimization", {})
    num_steps = opt_config.get("steps", 100)
    lr = opt_config.get("learning_rate", 0.05)
    image_size = opt_config.get("image_size", 256)
    
    # ### IMPROVEMENT: Memory Fix
    # Reduced default faces_per_pixel from 50 to 15 to prevent "Bin size too small" error.
    faces_per_pixel = opt_config.get("faces_per_pixel", 15) 
    
    blend_cfg = opt_config.get("blend_params", {})
    blend_params = BlendParams(sigma=blend_cfg.get("sigma", 1e-4), gamma=blend_cfg.get("gamma", 1e-4)) 
    
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=torch.log(torch.tensor(1. / 1e-4 - 1.)) * blend_params.sigma, 
        faces_per_pixel=faces_per_pixel, 
        # Optional: Explicitly increase bin capacity if meshes are very dense
        # max_faces_per_bin=10000 
    )
    
    # We initialize renderer ONCE, but we pass updated cameras in the loop
    # (Note: In Pytorch3D mesh renderer, you usually pass cameras to the call, not init, 
    # but here we follow your structure of passing it via Rasterizer update or forward)
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    
    optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=lr)
    
    # ### IMPROVEMENT: Learning Rate Scheduler
    # Reduces LR as we get closer to the solution to prevent jitter
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    
    # Loss weights
    loss_weights = config.get("loss_weights", {})
    w_vis = loss_weights.get("visibility", 10.0)
    w_center = 5.0 # Weight for new centering loss
    
    # Pre-compute meshgrid for centering loss
    y_coords = torch.linspace(-1, 1, image_size).to(device)
    x_coords = torch.linspace(-1, 1, image_size).to(device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    print(f"Beginning Camera Optimization (Steps: {num_steps})...")
    
    best_loss = float('inf')
    best_params = (3.5, 20.0, 0.0)

    for i in range(num_steps):
        optimizer.zero_grad()
        
        # 1. Get current camera and current physical values
        cameras, curr_dist, curr_elev = optimizer_model()
        
        # 2. Render
        reflection_image = renderer_silhouette(
            meshes_world=reflected_chair_mesh, 
            cameras=cameras
        )
        alpha_mask = reflection_image[..., 3] # (N, H, W)
        
        # 3. Calculate Losses
        
        # A. Visibility Loss (Maximize silhouette area)
        avg_alpha = torch.mean(alpha_mask)
        loss_visibility = -avg_alpha * w_vis
        
        # B. ### IMPROVEMENT: Centering Loss
        # Penalizes if the center of mass of the object is far from image center (0,0)
        total_alpha = torch.sum(alpha_mask) + 1e-6
        center_x = torch.sum(grid_x * alpha_mask) / total_alpha
        center_y = torch.sum(grid_y * alpha_mask) / total_alpha
        loss_centering = (center_x**2 + center_y**2) * w_center
        
        # C. Distance Target (Optional, soft pull towards target)
        # Note: We don't need Min/Max constraints here because the Sigmoid in the model handles it!
        target_dist = config.get("constraints", {}).get("target_distance", 3.5)
        loss_dist = (curr_dist - target_dist).abs()
        
        # Total Loss
        loss = loss_visibility + loss_centering + loss_dist
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        scheduler.step() # Update LR
        
        # Tracking
        current_val_dist = curr_dist.item()
        current_val_elev = curr_elev.item()
        current_val_azim = optimizer_model.azim.item()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = (current_val_dist, current_val_elev, current_val_azim)
        
        if i % 20 == 0:
            print(f" Step {i}: Dist={current_val_dist:.2f}, Elev={current_val_elev:.2f}, "
                  f"Loss={loss.item():.4f} (Vis: {loss_visibility.item():.3f}, Center: {loss_centering.item():.3f})")
            
    print(f"Optimization Complete. Best View: Dist={best_params[0]:.2f}, Elev={best_params[1]:.2f}, Azim={best_params[2]:.2f}")
            
    return best_params
