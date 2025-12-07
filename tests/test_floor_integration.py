
import unittest
import torch
import os
import shutil
from pathlib import Path
from pytorch3d.io import save_obj
from scene_composition.pytorch3d_scene_composition import SceneComposition
from depth_extraction.pytorch3d_depth_extractor import PyTorch3DDepthExtractor

class TestFloorIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_floor_test")
        self.test_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a dummy OBJ file
        self.obj_path = self.test_dir / "cube.obj"
        self._create_dummy_obj(self.obj_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_dummy_obj(self, path):
        # Create a simple cube using PyTorch3D and save it
        verts = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=torch.float32)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
            [3, 2, 6], [3, 6, 7], [0, 4, 5], [0, 5, 1]
        ], dtype=torch.int64)
        save_obj(path, verts, faces)

    def test_floor_composition_and_extraction(self):
        print(f"Testing on device: {self.device}")
        
        # 1. Test Scene Composition
        try:
            compositor = SceneComposition(device=self.device)
            scene = compositor.compose_scene([str(self.obj_path)])
            
            # Verify keys
            self.assertIn('objects', scene)
            self.assertIn('mirror', scene)
            self.assertIn('reflections', scene)
            self.assertIn('floor', scene, "Floor key missing from scene")
            
            # Verify list contents
            self.assertTrue(len(scene['floor']) > 0, "Floor list is empty")
            
            print("Scene composition successful. Floor present.")
            
        except Exception as e:
            self.fail(f"Scene composition failed: {e}")

        # 2. Test Depth Extraction
        try:
            extractor = PyTorch3DDepthExtractor(
                image_size=(256, 256),
                output_dir=str(self.test_dir),
                device=self.device
            )
            
            depth_map = extractor.extract_depth_map(scene, output_prefix="test_depth")
            
            self.assertTrue(os.path.exists(depth_map.image_path))
            print(f"Depth extraction successful. Saved to {depth_map.image_path}")
            
        except Exception as e:
            self.fail(f"Depth extraction failed: {e}")

if __name__ == '__main__':
    unittest.main()
