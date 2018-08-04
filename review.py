from matplotlib import pyplot as plt

from utils import *
from otsu_segmentation import *

files = {
    "jpg1": "testing_files/apple_healthy.jpg",
    "jpg2": "testing_files/apple_healthy_marked.jpg",
    "jpg3": "testing_files/jpg3.jpg",
    "jpg4": "testing_files/jpg4.jpg",
    "jpg5": "testing_files/jpg5.jpg",
    "jpg6": "testing_files/jpg6.jpg",
    "jpg7": "testing_files/jpg7.jpg",
    "jpg8": "testing_files/jpg8.jpg",
}

from background_marker import *


def show_review(original_image, image, image_title, hist_val=None, gray=False):
    if hist_val is None:
        plot_nums = 2
    else:
        plot_nums = 3

    cmap = 'gray' if gray else None
    # Original image plot
    original_image_show = original_image if gray else cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plot_index = 1
    plt.subplot(plot_nums, 1, plot_index), plt.imshow(original_image_show, cmap=cmap)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # Histogram plot
    if hist_val is not None:
        plot_index += 1
        plt.subplot(plot_nums, 1, plot_index), plt.hist(original_image.ravel(), 256)
        plt.axvline(x=hist_val, color='r', linestyle='dashed', linewidth=2)
        plt.title(image_title + ' Histogram'), plt.xticks([]), plt.yticks([])

    # Processed image plot
    image_show = image if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_index += 1
    plt.subplot(plot_nums, 1, plot_index), plt.imshow(image_show, cmap=cmap)
    plt.title(image_title), plt.xticks([]), plt.yticks([])

    print('plt showwing')
    plt.show()

def review_marker(file_name):
    try:
        original_image = read_image(file_name)            
        ret_val, marker = get_marker(original_image)
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)  
        else:
            raise
    else:
        show_review(original_image, marker, 'Otsu Thresholding Marker', ret_val)


def review_segmentation(file_name):
    try:
        original_image = read_image(file_name)        
        ret_val, segmented_image = segment_with_otsu(file_name)
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, segmented_image, 'Otsu Thresholding', ret_val, gray=True)


def review_remove_whites(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)
        remove_whites(original_image, marker)

        image = original_image.copy()
        debug(image, 'image')
        debug(marker, 'marker')
        image[np.logical_not(marker)] = np.array([0, 0, 0])
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Remove Reds')


def review_remove_blacks(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)
        remove_blacks(original_image, marker)

        image = original_image.copy()
        image[np.logical_not(marker)] = np.array([255, 255, 255])
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Remove Blacks')


def review_remove_blues(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)
        remove_blues(original_image, marker)

        image = original_image.copy()
        image[np.logical_not(marker)] = np.array([0, 0, 0])
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Remove Blues')


def review_excess_green(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        index = excess_green(original_image)
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, index, 'Green Index')


def review_excess_red(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        index = excess_red(original_image)
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, index, 'Red Index')


def review_excess_diff(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        index = index_diff(original_image)
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, index, 'Excess Diff')


def review_index_marker(file_name, contrast=False):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)
        color_index_marker(index_diff(original_image), marker)

        image = original_image.copy()
        image[np.logical_not(marker)] = np.array([0, 0, 0])
        if contrast:
            image[marker] = np.array([255, 255, 255])
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Index Marker')


def review_otsu_index(file_name):
    try:
        original_image = read_image(file_name)
        ret_val, image = otsu_color_index(excess_green(original_image), excess_red(original_image))

    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Otsu for Index', ret_val)


def review_texture_filter(file_name):
    try:
        original_image = read_image(file_name, cv2.IMREAD_GRAYSCALE)

        marker = np.full((original_image.shape[0], original_image.shape[1]), True)
        texture_filter(original_image, marker, threshold=280)

        image = original_image.copy()
        image[np.logical_not(marker)] = np.array([0])
        image[marker] = np.array([255])
    except ValueError as err:
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        show_review(original_image, image, 'Texture filter', gray=True)


def review_folder(folder):
    import os
    import re
    ext = re.compile('(\.jpe?g)|(\.png)$', re.I)
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            comm  = input('Continue: ')
            if comm == 'q':
                break
            if ext.search(file):
                print('file', os.path.join(folder,file))
                review_index_marker(os.path.join(folder,file))
            else:
                print('Warning: {} doesnt have valid image extension.'.format(file))


if __name__ == '__main__':
    while True:
        image_num = input("Enter image number: ").strip()

        file_name = files['jpg' + image_num]
        # review_marker(file_name)
        # review_segmentation(file_name)
        # review_remove_whites(file_name)
        # review_remove_blacks(file_name)
        # review_excess_green(file_name)
        # review_excess_red(file_name)
        review_index_marker(file_name)
        # review_texture_filter(file_name)