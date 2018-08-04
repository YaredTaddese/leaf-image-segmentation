import cv2
import numpy as np

from utils import *


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

def apply_marker(image, marker, background = 0, inverse = True):
    """
    Apply marker on original image
    Args:
        image: original image to be masked
        marker: one hot encoded with 0 and 255
        background: grayscale value that will be set for background
        inverse: if boolean should be inversed to avoid giving background as
                 output rather than the leaf parts

    Returns:
        new_image that is masked with marker
    """

    # change marker to boolean index
    # mask = np.logical_not(marker.astype(bool))
    mask = marker.astype(bool)
    unique, counts = np.unique(mask, return_counts=True)
    unique_counts = dict(zip(unique, counts))
    # needs enhancment here, if difference is not much inverse
    # so that it will not segment the background rather than the leaf
    # 2000000 number should be enchanced(found empirically, with 8 images, bad)
    if inverse and unique_counts[True] - unique_counts[False] < 2000000:
        mask = np.logical_not(mask)

    new_image = image.copy()
    new_image[mask] = background
    new_image[~mask] = 255

    return new_image

def segment_with_otsu(image_file, background = 0):
    """
    Segment an image file using otsu thresholding
    Args:
        image_file: file path
        background: grayscale value to be set as background

    Returns:
        ret_val:
        segmented_image: in ndarray form
    """
    image = read_image(image_file, cv2.IMREAD_GRAYSCALE)
    
    ret_val, marker = get_marker(image)
    segmented_image = apply_marker(image, marker, background)

    return ret_val, segmented_image
