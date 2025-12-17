import numpy as np

def interpolate_view_params(start_params, end_params, num_steps=19):
    """
    Generate a pool of camera poses by interpolating between two extreme poses.
    
    Args:
        start_params: Tuple of (distance, elevation, azimuth) for the start pose.
        end_params: Tuple of (distance, elevation, azimuth) for the end pose.
        num_steps: Number of total steps (poses) to generate.
        
    Returns:
        List of tuples, where each tuple is (distance, elevation, azimuth).
    """
    start_dist, start_elev, start_azim = start_params
    end_dist, end_elev, end_azim = end_params
    
    # Generate linearly spaced values for each parameter
    distances = np.linspace(start_dist, end_dist, num_steps)
    elevations = np.linspace(start_elev, end_elev, num_steps)
    azimuths = np.linspace(start_azim, end_azim, num_steps)
    
    # Combine into a list of tuples
    poses = []
    for i in range(num_steps):
        poses.append((float(distances[i]), float(elevations[i]), float(azimuths[i])))
        
    return poses
