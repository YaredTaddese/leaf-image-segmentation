import unittest
import sys
import os
import numpy as np

# include parent directory to get modules in there
sys.path.insert(0, os.path.pardir)

from image_segmentation import segmentation as seg

files = {
    "jpg": "tests/testing_files/jpg.jpg",
    "txt": "tests/testing_files/txt.txt",
}

class TestSegmentationUtils(unittest.TestCase):
    def test_can_read_jpg_file(self):
        image = seg.read_image(files["jpg"])
        self.assertEqual(type(image), np.ndarray)

    def test_cannot_read_txt_file(self):
        image = seg.read_image(files['txt'])
        self.assertEqual(image, None)

    def test_applying_marker(self):
        # fake ndarrays to mimic image and marker
        image = np.array([[1, 2, 3], [4, 5, 6]])
        marker = np.array([[0, 0, 255], [255, 0, 0]])

        new_image = seg.apply_marker(image, marker)
        np.testing.assert_array_equal(new_image, np.array([[1, 2, 0], [0, 5, 6]]))

if __name__ == '__main__':
    unittest.main()