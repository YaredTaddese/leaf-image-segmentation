import cv2
import numpy as np

from matplotlib import pyplot as plt

image_file = '../../Pictures/leafs/download (1).jpeg'
img = cv2.imread(image_file, 0)

plt.subplot(2,1,1), plt.imshow(img,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2), plt.hist(img.ravel(), 256)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
 
plt.show()