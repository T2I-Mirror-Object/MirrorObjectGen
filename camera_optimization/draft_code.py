import torch
import torch.nn as nn
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader,
    BlendParams,
    look_at_view_transform
)

class BestViewOptimizer(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. Learnable Parameters: Polar coordinates (distance, elevation, azimuth)
        # We initialize them to a "bad" random view to prove optimization works
        self.dist = nn.Parameter(torch.tensor(3.0))   
        self.elev = nn.Parameter(torch.tensor(10.0))  
        self.azim = nn.Parameter(torch.tensor(0.0))   
        
    def forward(self):
        # Convert polar params to Rotation (R) and Translation (T)
        R, T = look_at_view_transform(self.dist, self.elev, self.azim)
        
        # Update camera with new R, T
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        return cameras


# We use a Soft Silhouette Shader to allow gradients to flow
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=50, 
)

renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

def optimize_view(reflected_chair_mesh, mirror_frame_mesh, optimizer_model):
    # Setup optimizer (Adam usually works best for geometric params)
    optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=0.05)
    
    print("Beginning Camera Optimization...")
    
    for i in range(100): # 100 steps is usually plenty
        optimizer.zero_grad()
        
        # 1. Get current camera
        cameras = optimizer_model()
        
        # 2. Render only the REFLECTED CHAIR
        # Output shape: (1, 256, 256, 4) - Last channel is Alpha (0 to 1)
        reflection_image = renderer_silhouette(
            meshes_world=reflected_chair_mesh, 
            cameras=cameras
        )
        alpha_mask = reflection_image[..., 3] # Get silhouette
        
        # 3. Calculate Losses
        
        # A. Visibility Loss: We want more white pixels (sum of alpha)
        # We negate it because we want to MAXIMIZE visibility
        loss_visibility = -torch.sum(alpha_mask)
        
        # B. Centering Loss: The centroid of the reflection should be near image center
        # (Simple coordinate grid generation omitted for brevity)
        # loss_centering = ... 
        
        # C. Size Constraint: Don't let the camera get too close (clipping)
        # or too far (tiny reflection)
        loss_dist = (optimizer_model.dist - 3.5).abs() 

        # Total Loss
        loss = loss_visibility + (loss_dist * 100)
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Step {i}: Dist={optimizer_model.dist.item():.2f}, Loss={loss.item():.4f}")
            
    return optimizer_model.dist, optimizer_model.elev, optimizer_model.azim