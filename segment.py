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
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    # update marker to remove blues
    # remove_blues(original_image, marker)

    return original_image, marker


def segment_leaf(image_file, filling_mode, smooth_boundary, marker_intensity):
    """
    Segments leaf from an image file

    Args:
        image_file (string): full path of an image file
        filling_mode (string {no, flood, threshold, morph}): 
            how holes should be filled in segmented leaf
        smooth_boundary (boolean): should leaf boundary smoothed or not
        marker_intensity (int in rgb_range): should output background marker based
                                             on this intensity value as foreground value

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
                            or the segmented image based on marker_intensity value
    """
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    largest_mask = \
        select_largest_obj(bin_image, fill_mode=filling_mode,
                           smooth_boundary=smooth_boundary)

    if marker_intensity > 0:
        largest_mask[largest_mask != 0] = marker_intensity
        image = largest_mask
    else:
        # apply marker to original image
        image = original.copy()
        image[largest_mask == 0] = np.array([0, 0, 0])

    return original, image


def rgb_range(arg):
    """
    Check if arg is in range for rgb value(between 0 and 255)

    Args:
        arg (int convertible): value to be checked for validity of range

    Returns:
        arg in int form if valid

    Raises:
        argparse.ArgumentTypeError: if value can not be integer or not in valid range
    """

    try:
        value = int(arg)
    except ValueError as err:
       raise argparse.ArgumentTypeError(str(err))

    if value < 0 or value > 255:
        message = "Expected 0 <= value <= 255, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value


if __name__ == '__main__':
    # handle command line arguments
    parser = argparse.ArgumentParser('segment')
    parser.add_argument('-m', '--marker_intensity', type=rgb_range, default=0,
                        help='Output image will be as black background and foreground '
                             'with integer value specified here')
    parser.add_argument('-f', '--fill', choices=['no', 'flood', 'threshold', 'morph'],
                        help='Change hole filling technique for holes appearing in segmented output',
                        default='flood')
    parser.add_argument('-s', '--smooth', action='store_true',
                        help='Output image with smooth edges')
    parser.add_argument('-d', '--destination',
                        help='Destination directory for output image. '
                             'If not specified destination directory will be input image directory')
    parser.add_argument('image_source', help='A path of image filename or folder containing images')
    
    # set up command line arguments conveniently
    args = parser.parse_args()
    filling_mode = FILL[args.fill.upper()]
    smooth = True if args.smooth else False
    if args.destination:
        if not os.path.isdir(args.destination):
            print(args.destination, ': is not a directory')
            exit()

    # set up files to be segmented and destination place for segmented output
    if os.path.isdir(args.image_source):
        files = [entry for entry in os.listdir(args.image_source)
                 if os.path.isfile(os.path.join(args.image_source, entry))]
        base_folder = args.image_source

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            destination = args.image_source + '_markers'
            os.makedirs(destination, exist_ok=True)
    else:
        folder, file = os.path.split(args.image_source)
        files = [file]
        base_folder = folder

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            destination = folder

    for file in files:
        try:
            # read image and segment leaf
            original, output_image = \
                segment_leaf(os.path.join(base_folder, file), filling_mode, smooth, args.marker_intensity)

        except ValueError as err:
            if str(err) == IMAGE_NOT_READ:
                print('Error: Could not read image file: ', file)
            elif str(err) == NOT_COLOR_IMAGE:
                print('Error: Not color image file: ', file)
            else:
                raise
        # if no error when segmenting write segmented output
        else:
            # handle destination folder and fileaname
            filename, ext = os.path.splitext(file)
            new_filename = filename + '_marked' + ext
            new_filename = os.path.join(destination, new_filename)

            # write image to file
            cv2.imwrite(new_filename, output_image)
            print('Marker generated for image file: ', file)

