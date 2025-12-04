import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
# We need to be careful about imports in the script executing on import
# The script has if __name__ == "__main__", so it should be fine.
import scripts.depth_map_lgm_full as depth_map_script

class TestDepthMapLGMDryRun(unittest.TestCase):
    @patch('scripts.depth_map_lgm_full.LGMFull')
    @patch('scripts.depth_map_lgm_full.PointCloudSceneComposition')
    @patch('scripts.depth_map_lgm_full.PointCloudDepthExtractor')
    @patch('scripts.depth_map_lgm_full.TextParserImpl2')
    def test_generate_depth_for_prompt(self, mock_parser, mock_extractor, mock_compositor, mock_lgm):
        # Setup mocks
        mock_parser.return_value.parse.return_value = ["object1", "object2"]
        
        mock_lgm_instance = mock_lgm.return_value
        mock_lgm_instance.convert_multiple_texts_to_3d.return_value = ["path/to/obj1.ply", "path/to/obj2.ply"]
        
        mock_compositor_instance = mock_compositor.return_value
        mock_compositor_instance.compose_scene.return_value = {"objects": [], "mirror": [], "reflections": []}
        
        mock_extractor_instance = mock_extractor.return_value
        expected_output = "results/depth/scene_depth_lgm.png"
        mock_extractor_instance.extract_depth_map.return_value = expected_output
        
        # Run function
        result = depth_map_script.generate_depth_for_prompt(
            prompt="test prompt",
            output_dir="test_results"
        )
        
        # Verify
        mock_parser.return_value.parse.assert_called_with("test prompt")
        mock_lgm_instance.convert_multiple_texts_to_3d.assert_called()
        mock_compositor_instance.compose_scene.assert_called()
        mock_extractor_instance.extract_depth_map.assert_called()
        
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
