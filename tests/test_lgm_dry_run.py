import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_to_3d.lgm_full import LGMFull

class TestLGMFullDryRun(unittest.TestCase):
    @patch('text_to_3d.lgm_full.DiffusionPipeline')
    def test_convert_text_to_3d_flow(self, mock_pipeline):
        # Mock the pipeline
        mock_lgm = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_lgm
        
        # Setup return values
        mock_result = MagicMock()
        mock_lgm.return_value = mock_result
        
        # Initialize
        lgm = LGMFull(device="cpu")
        
        # Run conversion
        output_dir = "test_output"
        text = "a blue chair"
        result_path = lgm.convert_text_to_3d(text, output_dir)
        
        # Verify calls
        # 1. Check if pipeline was loaded
        mock_pipeline.from_pretrained.assert_called_once()
        
        # 2. Check if lgm was called with text and None
        mock_lgm.assert_called_with(text, None)
        
        # 3. Check if save_ply was called
        expected_path = os.path.join(output_dir, "a_blue_chair.ply")
        mock_lgm.save_ply.assert_called_with(mock_result, expected_path)
        
        self.assertEqual(result_path, expected_path)

if __name__ == '__main__':
    unittest.main()
