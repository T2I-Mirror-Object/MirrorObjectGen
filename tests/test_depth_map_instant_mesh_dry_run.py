import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.depth_map_instant_mesh as depth_map_script

class TestDepthMapInstantMeshDryRun(unittest.TestCase):
    @patch('scripts.depth_map_instant_mesh.InstantMesh')
    @patch('scripts.depth_map_instant_mesh.SceneComposition')
    @patch('scripts.depth_map_instant_mesh.PyTorch3DDepthExtractor')
    @patch('scripts.depth_map_instant_mesh.TextParserImpl2')
    def test_generate_depth_for_prompt(self, mock_parser, mock_extractor, mock_compositor, mock_im):
        # Setup mocks
        mock_parser.return_value.parse.return_value = ["object1", "object2"]
        
        mock_im_instance = mock_im.return_value
        mock_im_instance.convert_multiple_texts_to_3d.return_value = ["path/to/obj1.obj", "path/to/obj2.obj"]
        
        mock_compositor_instance = mock_compositor.return_value
        mock_compositor_instance.compose_scene.return_value = {"objects": [], "mirror": [], "reflections": []}
        
        mock_extractor_instance = mock_extractor.return_value
        expected_output = "results/depth/scene_depth_instant_mesh.png"
        mock_extractor_instance.extract_depth_map.return_value.image_path = expected_output
        
        # Run function
        result = depth_map_script.generate_depth_for_prompt(
            prompt="test prompt",
            output_dir="test_results"
        )
        
        # Verify
        mock_parser.return_value.parse.assert_called_with("test prompt")
        mock_im_instance.convert_multiple_texts_to_3d.assert_called()
        mock_compositor_instance.compose_scene.assert_called()
        mock_extractor_instance.extract_depth_map.assert_called()
        
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
