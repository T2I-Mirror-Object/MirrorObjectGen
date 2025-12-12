import cv2
import numpy as np

def extract_canny_edges(image_array: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """
    Extract Canny edges from an RGB image.
    
    Args:
        image_array: RGB image as numpy array (H, W, 3) or Grayscale (H, W)
        low_threshold: Lower bound for hysteresis thresholding
        high_threshold: Upper bound for hysteresis thresholding
        
    Returns:
        Edge map as numpy array (H, W) with values 0 or 255.
    """
    if image_array is None:
        raise ValueError("Input image is None")
        
    # Convert to grayscale if RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array
        
    # Ensure uint8
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
        
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    return edges
