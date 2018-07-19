import os
import argparse
import numpy as np
import cv2

from utils import *
from background_marker import *


def generate_background_marker(file):
    """
    Generate background marker for an image

    Args:
        file (string): full path of an image file

    Returns:
        tuple[0] (ndarray of an image): original image
        tuple[1] (ndarray size of an image): background marker
    """

    # check file name validity
    if not os.path.isfile(file):
        print(file, ': is not a file')
        return

    try:
        original_image = read_image(file)

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)

        # update marker based on vegetation color index technique
        color_index_marker(index_diff(original_image), marker)

        # update marker to remove blues
        # remove_blues(original_image, marker)

        return original_image, marker


    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Could not read image file: ', file)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file)
        else:
            raise


if __name__ == '__main__':
    # handle command line arguments
    parser = argparse.ArgumentParser('generate_marker')
    parser.add_argument('-c', '--contrast', action='store_true',
                        help='The image will be output as black background and white foreground')
    parser.add_argument('-f', '--fill', choices=['no', 'flood', 'threshold', 'morph'],
                        help='Change the hole filling technique')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='Output image with smooth edges')
    parser.add_argument('-d', '--destination',
                        help='Destination directory for the output image. '
                             'If not specified destination directory will be input image directory')
    parser.add_argument('image_file', help='An image filename with its full path')
    args = parser.parse_args()
    if args.fill is not None:
        filling_mode = FILL[args.fill.upper()]
    else:
        filling_mode = FILL['FLOOD']

    # get background marker and original image
    original, marker = generate_background_marker(args.image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    smooth = True if args.smooth else False
    largest_mask = \
        select_largest_obj(bin_image, fill_mode=filling_mode, smooth_boundary=smooth)

    if args.contrast:
        image = largest_mask
    else:
        # apply marker to original image
        image = original.copy()
        image[largest_mask == 0] = np.array([0, 0, 0])

    # handle destination folder and file
    filename, ext = os.path.splitext(args.image_file)
    new_filename = filename + '_marked' + ext
    if args.destination:
        if not os.path.isdir(args.destination):
            print(args.destination, ': is not a directory')
            exit()
        basename = os.path.basename(new_filename)
        new_filename = os.path.join(args.destination, basename)

    # write image to file
    cv2.imwrite(new_filename, image)
    print('Marker generated for image file: ', args.image_file)

