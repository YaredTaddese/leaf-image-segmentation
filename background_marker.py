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

    # below line same as: white_remover = np.logical_and(white_remover,  image[:, :, 0] > 200)
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
    # setup the black remover to process logical_and in place
    black_remover = np.full((image.shape[0], image.shape[1]), True)

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 0] < 30)
    black_remover[image[:, :, 0] >= 30] = False  # blue channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 1] < 30)
    black_remover[image[:, :, 1] >= 30] = False  # green channel

    # below line same as: black_remover = np.logical_and(black_remover,  image[:, :, 2] < 30)
    black_remover[image[:, :, 2] >= 30] = False  # red channel

    # remove blacks from marker
    marker[black_remover] = False


def remove_blues(image, marker):
    """
    Remove pixels resembling blues better than green from marker as background
    Args:
        image:
        marker: to be overloaded with blue pixels to be removed

    Returns:
        nothing
    """
    # choose pixels that have higher blue than green
    blue_remover = image[:, :, 0] > image[:, :, 1]

    # remove blues from marker
    marker[blue_remover] = False


def color_index_marker(color_index_diff, marker):
    """
    Differentiate marker based on the difference of the color indexes
    Threshold below some number(found empirically based on testing on 5 photos,bad)
    If threshold number is getting less, more non-green image
     will be included and vice versa
    Args:
        color_index_diff: color index difference based on green index minus red index
        marker: marker to be updated

    Returns:
        nothing
    """
    marker[color_index_diff <= -0.05] = False


def texture_filter(image, marker, threshold=220, window=3):
    window = window - window//2 - 1
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            # print('x y', x, y)
            # print('window', image[x:x + window, y:y + window])
            x_start = x - window if x < window else x
            y_start = y - window if y < window else y
            x_stop = x + window if x < image.shape[0] - window else image.shape[0]
            y_stop = y + window if y < image.shape[1] - window else image.shape[1]

            local_entropy = np.sum(image[x_start:x_stop, y_start:y_stop]
                                   * np.log(image[x_start:x_stop, y_start:y_stop] + 1e-07))
            # print('entropy', local_entropy)
            if local_entropy > threshold:
                marker[x, y] = False


def otsu_color_index(excess_green, excess_red):
    return cv2.threshold(excess_green - excess_red, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def generate_background_marker(file_name):
    image = read_image(file_name)

    marker = np.full([image.shape[0], image.shape[1]], True)

    index = excess_green(image) - excess_red(image)

    remove_whites(image, marker)
    remove_blacks(image, marker)
    remove_blues(image, marker)

    return 0, marker

def simple_test():
    # image = read_image(files['jpg1'])
    # g_img = excess_green(image)
    # r_img = excess_red(image)
    # debug(image[0], 'image')
    # debug(g_img[0], 'excess_green')
    # debug(r_img[0], 'excess_red')
    # debug(g_img[0]-r_img[0], 'diff')

    original_image = read_image(files['jpg1'], cv2.IMREAD_GRAYSCALE)
    marker = np.full((original_image.shape[0], original_image.shape[1]), True)
    texture_filter(original_image, marker)


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