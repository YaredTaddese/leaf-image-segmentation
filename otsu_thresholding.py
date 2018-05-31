import cv2
import numpy as np
from matplotlib import pyplot as plt

image_files = [
    '../../Pictures/leafs/images.jpeg',
    '../../Pictures/leafs/images (1).jpeg',
    '../../Pictures/leafs/images (3).jpeg',
    '../../Pictures/leafs/download (2).jpeg',
]
img = cv2.imread(image_files[0],0)

#ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
 
ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
#blur = cv2.GaussianBlur(img, (5,5), 0)
#ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
plt.subplot(3,1,1), plt.imshow(img,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2), plt.hist(img.ravel(), 256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.imshow(imgf,cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.show()