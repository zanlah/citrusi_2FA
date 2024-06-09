import unittest
from faceID import *

class TestMyScript(unittest.TestCase):

    def test_load_data(self):
        # Define a temporary directory with sample image files
        data_dir = "./tests/test_data"

        # Test the load_data function
        image_paths = load_data(data_dir)
        
        # Assert that the image_paths list is not empty
        self.assertTrue(len(image_paths) > 0)
        
        # Assert that each path in image_paths exists
        for path in image_paths:
            self.assertTrue(os.path.exists(path))

    def test_get_images_from_path(self):
        # Define a list of sample image paths
        image_paths = ["./tests/test_data/image1.png", "./tests/test_data/image2.png"]
        
        # Test the get_images_from_path function
        images = get_images_from_path(image_paths, 2)
        
        # Assert that the length of images is 2
        self.assertEqual(len(images), 2)
        
        # Assert that each element in images is an instance of PIL.Image.Image
        for img in images:
            self.assertIsInstance(img, Image.Image)

if __name__ == '__main__':
    unittest.main()
