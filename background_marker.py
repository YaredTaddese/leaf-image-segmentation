import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import *
from review import files


def remove_whites(image, marker):
    """
    Remove pixels resembling white from marker as background
    Args:
        image:
        marker: to be overloaded with white pixels to be removed

    Returns:
        nothing
    """
    # setup the white remover to process logical_and in place
    white_remover = np.full((image.shape[0], image.shape[1]), True)

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 0] > 220)
    white_remover[image[:, :, 0] <= 200] = False # blue channel

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 1] > 220)
    white_remover[image[:, :, 1] <= 220] = False  # green channel

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 2] > 200)
    white_remover[image[:, :, 2] <= 200] = False  # red channel

    # remove whites from marker
    marker[white_remover] = False


def remove_blacks(image, marker):
    """
    Remove pixels resembling black from marker as background
    Args:
        image:
        marker: to be overloaded with black pixels to be removed

    Returns:
        nothing
    """
    # generate the black remover
    black_remover = image[:, :, 0] < 30 # blue channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 1] < 30)
    black_remover[image[:, :, 1] >= 30] = False  # green channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 2] < 30)
    black_remover[image[:, :, 2] >= 30] = False  # red channel

    # remove blacks from marker
    marker[black_remover] = False


def diff_otsu(excess_green_image, excess_red_image):

    return cv2.threshold(excess_green_image - excess_red_image, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def generate_background_marker(file_name):
    image = read_image(file_name)

    marker = np.full([image.shape[0], image.shape[1]], True)

    excess_green_image = excess_green(image)
    excess_red_image = excess_red(image)

    remove_whites(image, marker)
    remove_blacks(image, marker)

    return 0, marker

def simple_test():
    image = read_image(files['jpg1'])
    g_img = excess_green(image)
    r_img = excess_red(image)
    debug(image[0], 'image')
    debug(g_img[0], 'excess_green')
    debug(r_img[0], 'excess_red')
    debug(g_img[0]-r_img[0], 'diff')
    print(type(image))


def test():

    image = read_image(files['jpg1'])

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)
    plt.show()

    # plt.imshow(cv2.cvtColor(excess_green(image), cv2.COLOR_BGR2RGB))
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(excess_red(image), cv2.COLOR_BGR2RGB))
    # plt.show()


if __name__ == '__main__':
    simple_test()