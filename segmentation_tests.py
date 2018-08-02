import unittest
import numpy as np

import otsu_segmentation as seg
import background_marker as bgm

files = {
    "jpg": "testing_files/apple_healthy_marked.jpg",
    "txt": "testing_files/txt.txt",
}

class TestSegmentationUtils(unittest.TestCase):
    def test_can_read_jpg_file(self):
        image = seg.read_image(files["jpg"])
        self.assertEqual(type(image), np.ndarray)

    def test_cannot_read_txt_file(self):
        with self.assertRaises(ValueError) as context:
            image = seg.read_image(files['txt'])
        self.assertEqual(seg.IMAGE_NOT_READ, str(context.exception))

    def test_applying_marker_with_no_inverse(self):
        # fake ndarrays to mimic image and marker
        image = np.array([[1, 2, 3], [4, 5, 6]])
        marker = np.array([[0, 0, 255], [255, 0, 0]])

        new_image = seg.apply_marker(image, marker, background=0, inverse=False)
        np.testing.assert_array_equal(new_image, np.array([[1, 2, 0], [0, 5, 6]]))
  
    def test_applying_marker_with_inverse(self):
        # fake ndarrays to mimic image and marker
        image = np.array([[1, 2, 3], [4, 5, 6]])
        marker = np.array([[0, 0, 255], [255, 0, 0]])

        new_image = seg.apply_marker(image, marker, background=0)
        np.testing.assert_array_equal(new_image, np.array([[0, 0, 3], [4, 0, 0]]))

    def test_remove_whites(self):
        # fake ndarrays to mimic image and marker
        image = np.array([
            [[200, 220, 200], [201, 221, 201], [100, 120, 100]],
            [[200, 120, 200], [201, 220, 201], [200, 221, 200]],
        ])
        marker = np.array([
            [True, True, True], [True, True, True]
        ])

        bgm.remove_whites(image, marker)
        np.testing.assert_array_equal(marker, np.array([ [True, False, True], [True, True, True] ]))


    def test_remove_blacks(self):
        # fake ndarrays to mimic image and marker
        image = np.array([
            [[30, 30, 30], [31, 31, 31], [29, 29, 29]],
            [[29, 29, 31], [29, 31, 31], [200, 221, 200]],
        ])
        marker = np.array([
            [True, True, True], [True, True, True]
        ])

        bgm.remove_blacks(image, marker)
        np.testing.assert_array_equal(marker, np.array([ [True, True, False], [True, True, True] ]))

if __name__ == '__main__':
    unittest.main()