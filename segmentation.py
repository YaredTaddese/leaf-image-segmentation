import cv2


def read_image(file_path):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file

    Returns:
        np.ndarray of the read image or None if couldn't read
    """
    # read image file in grayscale
    image = cv2.imread(file_path, 0)

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

def apply_marker(image, marker):
    """
    Apply marker on original image
    Args:
        image: original image to be masked
        marker: one hot encoded with 0 and 255

    Returns:
        new_image that is masked with marker
    """

    mask = marker.copy()
    mask[marker == 255] = 1

    new_image = image.copy()
    new_image[mask] = 0

    return new_image

def segment(image_file):
    image = read_image(image_file)
    if image is None:
        print('Error: Couldnot read image file: ', image_file)
    else:
        ret_val, marker = get_marker(image)
        segmented_image = apply_marker(image, marker)

        return ret_val, segmented_image