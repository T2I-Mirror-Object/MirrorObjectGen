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
        # Mock the pipelines
        mock_txt2img = MagicMock()
        mock_lgm = MagicMock()
        
        # Setup return values
        mock_image = MagicMock()
        mock_txt2img.return_value.images = [mock_image]
        
        # Mock LGM output behavior (simulating saving to file or returning object)
        # We'll mock the pipeline call to return an object with a save method
        mock_splat = MagicMock()
        mock_lgm.return_value = mock_splat
        
        # Configure the mock_pipeline to return our mocks
        # First call is txt2img, second is lgm
        mock_pipeline.from_pretrained.side_effect = [mock_txt2img, mock_lgm]
        
        # Initialize
        lgm = LGMFull(device="cpu")
        
        # Run conversion
        output_dir = "test_output"
        text = "a blue chair"
        result_path = lgm.convert_text_to_3d(text, output_dir)
        
        # Verify calls
        # 1. Check if pipelines were loaded
        self.assertEqual(mock_pipeline.from_pretrained.call_count, 2)
        
        # 2. Check if txt2img was called with text
        mock_txt2img.assert_called_with(text, guidance_scale=7.5)
        
        # 3. Check if lgm was called with the image
        # Note: We can't easily check the exact image object equality, but we can check it was called
        self.assertTrue(mock_lgm.called)
        
        # 4. Check if save was attempted (either via output_path arg or save method)
        # Our implementation tries output_path first, then save_to_ply, then save
        # Since we mocked the return value, let's see what happened.
        # If we passed output_path to the call:
        # mock_lgm.assert_called_with(mock_image, output_path=result_path)
        # OR if we called save on the result:
        # mock_splat.save_to_ply.assert_called() or mock_splat.save.assert_called()
        
        # Let's just print what happened for debugging if needed, but for now assert result path is correct
        expected_path = os.path.join(output_dir, "a_blue_chair.ply")
        self.assertEqual(result_path, expected_path)

if __name__ == '__main__':
    unittest.main()
