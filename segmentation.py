import cv2
import numpy as np

# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

def debug(value, name=None):
    if isinstance(value, np.ndarray):
        name = 'ndarray' if name is None else name

        print("{}: {}".format(name, value))
        print("{} shape: {}".format(name, value.shape))
    else:
        name = 'value' if name is None else name

        print("{}: {}".format(name, value))

def read_image(file_path):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file

    Returns:
        np.ndarray of the read image or None if couldn't read
    
    Raises:
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path,  cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image

def get_marker(image):
    """
    Get image marker to differentiate image background from foreground
    Args:
        image: image for which marker will be generated for

    Returns:
        ret_val:
        image_marker: marker
    """
    ret_val, image_marker = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ret_val, image_marker

def apply_marker(image, marker, background = 0):
    """
    Apply marker on original image
    Args:
        image: original image to be masked
        marker: one hot encoded with 0 and 255
        background: grayscale value that will be set for background

    Returns:
        new_image that is masked with marker
    """

    # change marker to boolean index
    # mask = np.logical_not(marker.astype(bool))
    mask = marker.astype(bool)

    new_image = image.copy()
    new_image[mask] = background

    return new_image

def segment(image_file, background = 0):
    """
    Segment an image file using otsu thresholding
    Args:
        image_file: file path
        background: grayscale value to be set as background

    Returns:
        ret_val:
        segmented_image: in ndarray form
    """
    image = read_image(image_file)
    
    ret_val, marker = get_marker(image)
    segmented_image = apply_marker(image, marker, background)

    return ret_val, segmented_image
