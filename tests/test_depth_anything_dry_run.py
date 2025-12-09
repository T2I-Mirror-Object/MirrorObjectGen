import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.depth_estimation import DepthAnything3Estimator

class TestDepthAnythingDryRun(unittest.TestCase):
    @patch('utils.depth_estimation.DepthAnything3')
    def test_initialization_and_inference(self, mock_da3_class):
        # Setup mock
        mock_model = MagicMock()
        mock_da3_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Setup mock prediction return
        mock_prediction = MagicMock()
        mock_prediction.depth = torch.randn(1, 480, 640)
        mock_prediction.conf = torch.randn(1, 480, 640)
        mock_prediction.extrinsics = torch.randn(1, 3, 4)
        mock_prediction.intrinsics = torch.randn(1, 3, 3)
        mock_model.inference.return_value = mock_prediction

        # Initialize estimator
        estimator = DepthAnything3Estimator(device=torch.device("cpu"))
        
        # Verify initialization
        mock_da3_class.from_pretrained.assert_called_with("depth-anything/da3mono-large")
        mock_model.to.assert_called()
        
        # Test inference with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.extract_depth(dummy_image)
        
        # Verify inference call
        mock_model.inference.assert_called_with(
            [dummy_image],
            export_dir=None,
            export_format="npz"
        )
        
        # Check result
        self.assertEqual(result, mock_prediction)
        self.assertEqual(result.depth.shape, (1, 480, 640))

if __name__ == '__main__':
    unittest.main()
