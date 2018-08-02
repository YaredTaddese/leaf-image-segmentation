import numpy as np
import cv2


# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

# error message when image is not colored while it should be
NOT_COLOR_IMAGE = 'NOT_COLOR_IMAGE'


def read_image(file_path, read_mode=cv2.IMREAD_COLOR):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file
        read_mode: whether image reading mode is rgb, grayscale or somethin

    Returns:
        np.ndarray of the read image or None if couldn't read

    Raises:
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path, read_mode)

    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image


def ensure_color(image):
    """
    Ensure that an image is colored
    Args:
        image: image to be checked for

    Returns:
        nothing

    Raises:
        ValueError with message code if image is not colored
    """
    if len(image.shape) < 3:
        raise ValueError(NOT_COLOR_IMAGE)


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.true_divide(a, b)
        q[ ~ np.isfinite(q) ] = 0  # -inf inf NaN

    return q


def excess_green(image, scale = 2):
    """
    Compute excess green index for colored image

    Args:
        image: image to be converted
        scale: number to scale green channel of the image

    Returns:
        new image with excess green
    """

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)
    debug(bgr_sum, 'green bgr sum')

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    index = scale * greens - (reds + blues)

    return index


def excess_red(image, scale=1.4):
    """
    Compute excess red index for colored image

    Args:
        image: image to be converted
        scale: number to scale red channel of the image

    Returns:
        new image with excess red
    """

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    index = scale * reds - greens

    return index


def index_diff(image, green_scale=2.0, red_scale=1.4):

    ensure_color(image)

    bgr_sum = np.sum(image, axis=2)

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    green_index = green_scale * greens - (reds + blues)
    red_index = red_scale * reds - (greens)

    return green_index - red_index


def debug(value, name=None):
    if isinstance(value, np.ndarray):
        name = 'ndarray' if name is None else name

        print("{}: {}".format(name, value))
        print("{} shape: {}".format(name, value.shape))
    else:
        name = 'value' if name is None else name

        print("{} type: {}".format(name, type(value)))
        print("{}: {}".format(name, value))

