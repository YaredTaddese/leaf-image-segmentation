import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

from utils import *
from review import files


# constants for filling holes mode
FILL = {
    'NO': 0,
    'FLOOD': 1,
    'THRESHOLD': 2,
    'MORPH': 3,
}


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
    """
    Update marker based on texture of an image

    Args:
        image (ndarray of grayscale image):
        marker (ndarray size of image): marker to be updated
        threshold (number): minimum size of texture measurement(entropy) allowed
        window (int): window size of a square the texture computed from

    Returns: nothing
    """

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


def generate_floodfill_mask(bin_image):
    """
    Generate a mask to remove backgrounds adjacent to image edge

    Args:
        bin_image (ndarray of grayscale image): image to remove backgrounds from

    Returns:
        a mask to backgrounds adjacent to image edge
    """
    y_mask = np.full(
        (bin_image.shape[0], bin_image.shape[1]), fill_value=255, dtype=np.uint8
    )
    x_mask = np.full(
        (bin_image.shape[0], bin_image.shape[1]), fill_value=255, dtype=np.uint8
    )

    xs, ys = bin_image.shape[0], bin_image.shape[1]

    for x in range(0, xs):
        item_indexes = np.where(bin_image[x,:] != 0)[0]
        # min_start_edge = ys
        # max_final_edge = 0
        if len(item_indexes):
            start_edge, final_edge = item_indexes[0], item_indexes[-1]
            x_mask[x, start_edge:final_edge] = 0
            # if start_edge < min_start_edge:
            #     min_start_edge = start_edge
            # if final_edge > max_final_edge:
            #     max_final_edge = final_edge

    for y in range(0, ys):
        item_indexes = np.where(bin_image[:,y] != 0)[0]
        if len(item_indexes):
            start_edge, final_edge = item_indexes[0], item_indexes[-1]

            y_mask[start_edge:final_edge, y] = 0
            # mask[:start_edge, y] = 255
            # mask[final_edge:, y] = 255

    return np.logical_or(x_mask, y_mask)


def select_largest_obj(img_bin, lab_val=255, fill_mode=FILL['FLOOD'],
                       smooth_boundary=False, kernel_size=15):
    """
    Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest
                object. Default is 255.
        fill_mode (string {no,flood,threshold,morph}): hole filling techniques which are
            - no: no filling of holes
            - flood: floodfilling technique without removing image edge sharing holes
            - threshold: removing holes based on minimum size of hole to be removed
            - morph: closing morphological operation with some kernel size to remove holes
        smooth_boundary ([boolean]): whether smooth the boundary of the
                largest object using morphological opening or not. Default
                is false.
        kernel_size ([int]): the size of the kernel used for morphological
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    """

    # set up components
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)

    # find largest component label(label number works with labeled image because of +1)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1

    # create a mask that will only cover the largest component
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val

    if fill_mode == FILL['FLOOD']:
        # fill holes using opencv floodfill function

        # set up seedpoint(starting point) for floodfill
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])

        # copied image to be floodfill
        img_floodfill = largest_mask.copy()

        # create a mask to ignore what shouldn't be filled(I think no effect)
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)

        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                    newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.

        # get a mask to avoid filling non-holes that are adjacent to image edge
        non_holes_mask = generate_floodfill_mask(largest_mask)
        holes_mask = np.bitwise_and(holes_mask, np.bitwise_not(non_holes_mask))

        largest_mask = largest_mask + holes_mask
    elif fill_mode == FILL['MORPH']:
        # fill holes using closing morphology operation
        kernel_ = np.ones((50, 50), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE,
                                        kernel_)
    elif fill_mode == FILL['THRESHOLD']:
        # fill background-holes based on hole size threshold
        # default hole size threshold is some percentage
        #   of size of the largest component(i.e leaf component)

        # invert to setup holes of background, sorry for the incovenience
        inv_img_bin = np.bitwise_not(largest_mask)

        # set up components
        inv_n_labels, inv_img_labeled, inv_lab_stats, _ = \
            cv2.connectedComponentsWithStats(inv_img_bin, connectivity=8, ltype=cv2.CV_32S)

        # find largest component label(label number works with labeled image because of +1)
        inv_largest_obj_lab = np.argmax(inv_lab_stats[1:, 4]) + 1

        # setup sizes and number of components
        inv_sizes = inv_lab_stats[1:, -1]
        sizes = lab_stats[1:, -1]
        inv_nb_components = inv_n_labels - 1

        # find the greater side of the image
        inv_max_side = np.amax(inv_img_labeled.shape)

        # set the minimum size of hole that is allowed to stay
        inv_min_size = int(0.3 * sizes[largest_obj_lab - 1]) # todo: specify good min size

        # generate the mask that allows holes greater than minimum size(weird)
        inv_mask = np.zeros((inv_img_labeled.shape), dtype=np.uint8)
        for inv_i in range(0, inv_nb_components):
            if inv_sizes[inv_i] >= inv_min_size:
                inv_mask[inv_img_labeled == inv_i + 1] = 255

        largest_mask = largest_mask + np.bitwise_not(inv_mask)

    if smooth_boundary:
        # smooth edge boundary
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN,
                                        kernel_)

    return largest_mask


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