import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene_composition.pytorch3d_scene_composition import SceneComposition
from segmentation_extraction.pytorch3d_segmentation_extractor import PyTorch3DSegmentationExtractor

class TestMirrorMaskExtraction(unittest.TestCase):
    @patch('scene_composition.pytorch3d_scene_composition.load_obj')
    @patch('scene_composition.pytorch3d_scene_composition.Meshes')
    def test_end_to_end_flow(self, mock_meshes, mock_load_obj):
        # 1. Setup minimal mocks
        device = "cpu"
        
        # Mock load_obj to return dummy data
        mock_verts = torch.zeros((10, 3))
        mock_faces = MagicMock()
        mock_faces.verts_idx = torch.zeros((10, 3), dtype=torch.int64)
        mock_load_obj.return_value = (mock_verts, mock_faces, None)
        
        # Mock Meshes to act like a list of verts when iterating or checking bounds
        mock_mesh_instance = MagicMock()
        mock_mesh_instance.verts_packed.return_value = torch.tensor([
            [-1., -1., 0.], [1., 1., 0.] # Bounds min=-1, max=1
        ])
        mock_mesh_instance.faces_list.return_value = [torch.zeros((1, 3), dtype=torch.int64)]
        mock_mesh_instance.num_faces_per_mesh.return_value = torch.tensor([1])
        mock_mesh_instance.to.return_value = mock_mesh_instance
        mock_mesh_instance.device = torch.device(device)
        
        # When creating meshes (box, etc.), return our mock instance
        mock_meshes.return_value = mock_mesh_instance
        
        # 2. Scene Composition
        compositor = SceneComposition(device=device)
        
        # We need to mock _load_objects effectively or just bypass it?
        # Let's bypass actual file loading by mocking _load_objects
        with patch.object(compositor, '_load_objects', return_value=[mock_mesh_instance]):
            scene = compositor.compose_scene(["dummy_obj.obj"])
            
        # Verify scene has 'mirror_surface'
        print("Scene keys:", scene.keys())
        self.assertIn('mirror_surface', scene)
        self.assertTrue(len(scene['mirror_surface']) > 0)
        
        # 3. Extraction
        extractor = PyTorch3DSegmentationExtractor(
            image_size=(128, 128),
            output_dir="test_results/seg",
            device=device
        )
        
        # Mock the renderer to avoid actual rasterization on CPU/GPU in unit test if possible,
        # OR just let it run if dependencies allow (PyTorch3D on CPU is fine).
        # But we don't have real geometry in the mocks, so rasterization might be empty.
        # Let's mock the renderer inside extractor.
        
        with patch('segmentation_extraction.pytorch3d_segmentation_extractor.InstanceIDRenderer') as mock_renderer_cls:
            mock_renderer = mock_renderer_cls.return_value
            # Return a dummy ID map (128x128)
            mock_renderer.return_value = torch.ones((1, 128, 128), dtype=torch.int32) # All pixels ID 1
            mock_renderer.to.return_value = mock_renderer
            
            output_path = extractor.extract_mirror_mask(scene, output_filename="test_mask.png")
            
            # Verify call
            mock_renderer_cls.assert_called()
            args, kwargs = mock_renderer.call_args
            # First arg should be the mirror surface mesh list
            self.assertEqual(args[0], scene['mirror_surface'])
            self.assertEqual(kwargs['mesh_to_id'], [1])
            
            # Verify output file
            print(f"Output path: {output_path}")
            self.assertIsNotNone(output_path)
            self.assertTrue(Path(output_path).exists())
            
            # Additional path check
            expected_dir = Path("test_results/mirror_mask")
            self.assertEqual(Path(output_path).parent, expected_dir)

if __name__ == '__main__':
    unittest.main()
