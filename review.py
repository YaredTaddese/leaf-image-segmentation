from segmentation import *
from matplotlib import pyplot as plt

files = {
    "jpg1": "tests/testing_files/jpg.jpg",
    "jpg2": "tests/testing_files/jpg2.jpg",
    "jpg3": "tests/testing_files/jpg3.jpg",
    "jpg4": "tests/testing_files/jpg4.jpg",
    "jpg5": "tests/testing_files/jpg5.jpg",
    "jpg6": "tests/testing_files/jpg6.jpg",
    "jpg7": "tests/testing_files/jpg7.jpg",
    "jpg8": "tests/testing_files/jpg8.jpg",
}

def review_marker(file_name):
    try:
        original_image = read_image(file_name)            
        ret_val, marker = get_marker(original_image)
    except ValueError as err:
        if err.message == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)  
        else:
            raise
    else:      
        # Original image plot
        plt.subplot(3, 1, 1), plt.imshow(original_image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        # Histogram plot
        plt.subplot(3, 1, 2), plt.hist(original_image.ravel(), 256)
        plt.axvline(x=ret_val, color='r', linestyle='dashed', linewidth=2)
        plt.title('Original Image Histogram'), plt.xticks([]), plt.yticks([])

        # Marker image plot
        plt.subplot(3, 1, 3), plt.imshow(marker, cmap='gray')
        plt.title('Otsu thresholding Marker'), plt.xticks([]), plt.yticks([])

        plt.show()


def review_segmentation(file_name):
    try:
        original_image = read_image(file_name)        
        ret_val, segmented_image = segment(file_name)
    except ValueError as err:
        if err.message == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        else:
            raise
    else:
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
    while True:
        image_num = input("Enter image number: ")

        file_name = files['jpg' + image_num]
        review_marker(file_name)
        review_segmentation(file_name)