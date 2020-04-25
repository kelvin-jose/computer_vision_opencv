import cv2
import numpy as np

image = cv2.imread('image.jpg', 0)

"""
a kernel slides over the binary image like convolution and
changes the current pixel value to 0 if all the pixels under
the kernel are not 1s i.e erode the pixels. So the pixels of
the edges of white region might change from 1 to 0.

"""

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(image, kernel, iterations=1)
cv2.imwrite('erosion.jpg', erosion)
