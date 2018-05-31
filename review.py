from segmentation import *
from matplotlib import pyplot as plt

files = {
    "jpg": "tests/testing_files/jpg.jpg",
    "jpg2": "tests/testing_files/jpg2.jpg",
    "jpg3": "tests/testing_files/jpg3.jpeg",
}

def review_marker():
    file_name = files['jpg3']
    original_image = read_image(file_name)
    ret_val, marker = get_marker(original_image)

    # Original image plot
    plt.subplot(1, 3, 1), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # Histogram plot
    plt.subplot(1, 3, 2), plt.hist(original_image.ravel(), 256)
    plt.axvline(x=ret_val, color='r', linestyle='dashed', linewidth=2)
    plt.title('Original Image Histogram'), plt.xticks([]), plt.yticks([])

    # Marker image plot
    plt.subplot(1, 3, 3), plt.imshow(marker, cmap='gray')
    plt.title('Otsu thresholding marker'), plt.xticks([]), plt.yticks([])

    plt.show()


def review_segmentation():
    file_name = files['jpg3']
    original_image = read_image(file_name)
    ret_val, segmented_image = segment(file_name)

    # Original image plot
    plt.subplot(3, 1, 1), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # Histogram plot
    plt.subplot(3, 1, 2), plt.hist(original_image.ravel(), 256)
    plt.axvline(x=ret_val, color='r', linestyle='dashed', linewidth=2)
    plt.title('Histogram'), plt.xticks([]), plt.yticks([])

    # Segmented image plot
    plt.subplot(3, 1, 3), plt.imshow(segmented_image, cmap='gray')
    plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    review_marker()
    review_segmentation()