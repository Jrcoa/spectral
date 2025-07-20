import unittest
import numpy as np
import os
from construct_dataset import build_target_image
from src.utils import load_image, parse_points_file

class TestConstructDataset(unittest.TestCase):

    # Replace load_image and parse_points_file with functions that return dummy data to test build_target_image
    def setUp(self):
        # These paths are placeholders
        self.mock_hdr_file = './tile_raster/Tile_Hyperspectral/Tile_Hyperspectral/tile_B2.hdr'
        self.mock_points_file = './trees_points.xlsx'
        
        def mock_load_image(hdr_file):
            return np.zeros((100, 100, 288)), None
        def mock_parse_points_file(points_file, map_info):
            return [(10, 10), (20, 20)], ['tree1', 'tree2']
        
        self.original_load_image = load_image
        self.original_parse_points_file = parse_points_file
        
        globals()['load_image'] = mock_load_image
        globals()['parse_points_file'] = mock_parse_points_file

    # Restore the state of the functions after the test
    def tearDown(self):
        globals()['load_image'] = self.original_load_image
        globals()['parse_points_file'] = self.original_parse_points_file

    def test_target_image_size(self):
        target_image = build_target_image(self.mock_hdr_file, self.mock_points_file)
        self.assertEqual(target_image.shape, (100, 100))

if __name__ == '__main__':
    unittest.main()