import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_to_3d.instant_mesh import InstantMesh

class TestInstantMeshDryRun(unittest.TestCase):
    @patch('text_to_3d.instant_mesh.DiffusionPipeline')
    @patch('text_to_3d.instant_mesh.instantiate_from_config')
    @patch('text_to_3d.instant_mesh.OmegaConf')
    @patch('text_to_3d.instant_mesh.hf_hub_download')
    @patch('text_to_3d.instant_mesh.save_obj_with_mtl')
    @patch('text_to_3d.instant_mesh.remove_background')
    @patch('text_to_3d.instant_mesh.resize_foreground')
    def test_convert_text_to_3d_flow(self, mock_resize, mock_rembg, mock_save, mock_hf, mock_omega, mock_instantiate, mock_pipeline):
        # Mock pipelines
        mock_txt2img = MagicMock()
        mock_zero123 = MagicMock()
        mock_pipeline.from_pretrained.side_effect = [mock_txt2img, mock_zero123]
        
        # Mock txt2img output
        mock_image = MagicMock()
        mock_txt2img.return_value.images = [mock_image]
        
        # Mock zero123 output
        mock_mv_images = MagicMock()
        # Need to mock numpy conversion behavior
        # The code does: np.asarray(mv_images, ...)
        # So mv_images should be list of images or similar
        # Let's mock it as a list of 1 element which is an image
        mock_mv_image = MagicMock()
        mock_zero123.return_value.images = [mock_mv_image]
        
        # Mock InstantMesh model
        mock_model = MagicMock()
        mock_instantiate.return_value = mock_model
        
        # Mock model outputs
        mock_planes = MagicMock()
        mock_model.forward_planes.return_value = mock_planes
        
        # Mock extract_mesh output: vertices, faces, uvs, mesh_tex_idx, tex_map
        mock_model.extract_mesh.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )

        # Initialize
        im = InstantMesh(device="cpu")
        
        # Run conversion
        output_dir = "test_output"
        text = "a blue chair"
        
        # We need to mock numpy array creation from the mocked images
        with patch('numpy.asarray', return_value=np.zeros((960, 640, 3), dtype=np.uint8)):
            result_path = im.convert_text_to_3d(text, output_dir)
        
        # Verify calls
        self.assertEqual(mock_pipeline.from_pretrained.call_count, 2)
        mock_txt2img.assert_called()
        mock_zero123.assert_called()
        mock_model.forward_planes.assert_called()
        mock_model.extract_mesh.assert_called()
        mock_save.assert_called()
        
        expected_path = os.path.join(output_dir, "a_blue_chair.obj")
        self.assertEqual(result_path, expected_path)

if __name__ == '__main__':
    unittest.main()
