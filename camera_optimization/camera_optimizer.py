import torch
import torch.nn as nn
import yaml
from pathlib import Path
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

def load_config(config_path="configs/camera-optimization.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class BestViewOptimizer(nn.Module):
    def __init__(self, device="cuda", config=None):
        super().__init__()
        self.device = device
        
        # Load defaults from config if available, else use hardcoded fallback
        if config:
            init_params = config.get("initial_parameters", {})
            dist_val = init_params.get("distance", 3.5)
            elev_val = init_params.get("elevation", 20.0)
            azim_val = init_params.get("azimuth", 0.0)
        else:
            dist_val = 3.5
            elev_val = 20.0
            azim_val = 0.0
        
        # 1. Learnable Parameters: Polar coordinates (distance, elevation, azimuth)
        self.dist = nn.Parameter(torch.tensor(float(dist_val)))   
        self.elev = nn.Parameter(torch.tensor(float(elev_val)))  
        self.azim = nn.Parameter(torch.tensor(float(azim_val)))   
        
    def forward(self):
        # Convert polar params to Rotation (R) and Translation (T)
        R, T = look_at_view_transform(self.dist, self.elev, self.azim)
        
        # Update camera with new R, T
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        return cameras

def optimize_view(scene, device='cuda', config_path="configs/camera-optimization.yaml"):
    """
    Optimizes camera parameters to get the best view of the reflection.
    
    Args:
        scene: Dictionary containing 'reflections' list of meshes.
        device: Device to run optimization on.
        config_path: Path to the configuration YAML file.
        
    Returns:
        Tuple[float, float, float]: Optimized (distance, elevation, azimuth)
    """
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Falling back to default values.")
        config = {}

    # Extract just the reflections to optimize for visibility
    reflection_meshes = scene.get('reflections', [])
    if not reflection_meshes:
        print("Warning: No reflection meshes found to optimize. Using default view.")
        init_params = config.get("initial_parameters", {})
        return (
            init_params.get("distance", 3.5),
            init_params.get("elevation", 20.0),
            init_params.get("azimuth", 0.0)
        )
        
    reflected_chair_mesh = join_meshes_as_scene(reflection_meshes).to(device)
    
    # Initialize model
    optimizer_model = BestViewOptimizer(device=device, config=config).to(device)
    
    # Get optimization params from config
    opt_config = config.get("optimization", {})
    num_steps = opt_config.get("steps", 100)
    lr = opt_config.get("learning_rate", 0.05)
    image_size = opt_config.get("image_size", 256)
    faces_per_pixel = opt_config.get("faces_per_pixel", 50)
    
    blend_cfg = opt_config.get("blend_params", {})
    sigma = blend_cfg.get("sigma", 1e-4)
    gamma = blend_cfg.get("gamma", 1e-4)
    
    # We use a Soft Silhouette Shader to allow gradients to flow
    blend_params = BlendParams(sigma=sigma, gamma=gamma) 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=torch.log(torch.tensor(1. / 1e-4 - 1.)) * blend_params.sigma, 
        faces_per_pixel=faces_per_pixel, 
    )
    
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=optimizer_model(), # Initial camera
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=lr)
    
    # Loss weights and constraints
    loss_weights = config.get("loss_weights", {})
    w_vis = loss_weights.get("visibility", 10.0)
    w_dist = loss_weights.get("distance", 1.0)
    w_elev_min = loss_weights.get("elevation_min", 1.0)
    w_elev_max = loss_weights.get("elevation_max", 1.0)
    
    constraints = config.get("constraints", {})
    target_dist = constraints.get("target_distance", 3.5)
    max_elev = constraints.get("max_elevation", 60.0)
    
    print("Beginning Camera Optimization...")
    
    for i in range(num_steps):
        optimizer.zero_grad()
        
        # 1. Get current camera
        cameras = optimizer_model()
        
        # 2. Render only the REFLECTED objects
        reflection_image = renderer_silhouette(
            meshes_world=reflected_chair_mesh, 
            cameras=cameras
        )
        alpha_mask = reflection_image[..., 3] # Get silhouette
        
        # 3. Calculate Losses
        
        # A. Visibility Loss: We want more white pixels (sum of alpha)
        # Normalize by image size to keep loss scale reasonable
        loss_visibility = -torch.sum(alpha_mask) / (image_size * image_size)
        
        # B. Distance Constraint
        loss_dist = (optimizer_model.dist - target_dist).abs()
        
        # C. Elevation Constraint
        loss_elev_min = torch.nn.functional.relu(-optimizer_model.elev) # Penalize < 0
        loss_elev_max = torch.nn.functional.relu(optimizer_model.elev - max_elev) # Penalize > max
        
        # Total Loss
        loss = (loss_visibility * w_vis) + (loss_dist * w_dist) + (loss_elev_min * w_elev_min) + (loss_elev_max * w_elev_max)
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"  Step {i}: Dist={optimizer_model.dist.item():.2f}, Elev={optimizer_model.elev.item():.2f}, Loss={loss.item():.4f}")
            
    best_dist = optimizer_model.dist.item()
    best_elev = optimizer_model.elev.item()
    best_azim = optimizer_model.azim.item()
    
    print(f"Optimization Complete. Best View: Dist={best_dist:.2f}, Elev={best_elev:.2f}, Azim={best_azim:.2f}")
            
    return best_dist, best_elev, best_azim
