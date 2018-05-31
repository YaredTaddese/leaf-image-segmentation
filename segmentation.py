import cv2


def read_image(file_path):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file

    Returns:
        np.ndarray of the read image
    """
    # read image file in grayscale
    image = cv2.imread(file_path, 0)
    if image is None:
        print('Error: Could not read image file: ', file_path)

    return image