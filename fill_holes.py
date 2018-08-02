import cv2
import numpy as np
from utils import *

img = cv2.imread('/home/yared/Documents/leaf images/Apple___healthy_markers/0bb2ddc5-d1f4-4fc2-be6b-6b63c60790df___RS_HL 7550.JPG',0)
img[img > 0 ] = 255
img_inv = cv2.bitwise_not(img)
debug(img_inv, 'img_inv')
print('type', img_inv.dtype)
nb_components, output, stats, centroids = \
    cv2.connectedComponentsWithStats(img_inv, connectivity=8, ltype=cv2.CV_32S)
# debug(nb_components, 'nb_components')
# debug(output, 'output')
# debug(stats, 'stats')
# debug(centroids, 'centroids')
sizes = stats[1:, -1]; nb_components = nb_components - 1
min_size = 300

img2 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

cv2.imshow('ImageWindow',img2)
# cv2.waitKey(0)