from matplotlib import pyplot as plt

from utils import *
from otsu_segmentation import *

files = {
    "jpg1": "testing_files/jpg.jpg",
    "jpg2": "testing_files/jpg2.jpg",
    "jpg3": "testing_files/jpg3.jpg",
    "jpg4": "testing_files/jpg4.jpg",
    "jpg5": "testing_files/jpg5.jpg",
    "jpg6": "testing_files/jpg6.jpg",
    "jpg7": "testing_files/jpg7.jpg",
    "jpg8": "testing_files/jpg8.jpg",
}

from background_marker import *

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

def review_remove_whites(file_name):
    try:
        original_image = read_image(file_name)
        ret_val = 0

        marker = np.full([original_image.shape[0], original_image.shape[1]], True)
        marker = remove_whites(original_image, marker)

        image = original_image.copy()
        image[np.logical_not(marker)] = [0, 0, 0]
    except ValueError as err:
        debug(err, 'err')
        if str(err) == IMAGE_NOT_READ:
            print('Error: Couldnot read image file: ', file_name)
        elif str(err) == NOT_COLOR_IMAGE:
            print('Error: Not color image file: ', file_name)
        else:
            raise
    else:
        # Original image plot
        plt.subplot(3, 1, 1), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        # Histogram plot
        plt.subplot(3, 1, 2), plt.hist(original_image.ravel(), 256)
        plt.axvline(x=ret_val, color='r', linestyle='dashed', linewidth=2)
        plt.title('Histogram'), plt.xticks([]), plt.yticks([])

        # Segmented image plot
        plt.subplot(3, 1, 3), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Remove whites'), plt.xticks([]), plt.yticks([])

        plt.show()

if __name__ == '__main__':
    while True:
        image_num = input("Enter image number: ")

        file_name = files['jpg' + image_num]
        # review_marker(file_name)
        # review_segmentation(file_name)
        review_remove_whites(file_name)