import cv2
import numpy as np

image = cv2.imread('image.jpg', 0)
kernel = np.ones((5, 5), np.uint8)

"""

This is kind of opposite of erosion. The algorithm
reverts the pixel if at-least one of the pixel under
the kernel is 1. So width of the foreground image
would increase. 

"""

dilation = cv2.dilate(image, kernel, iterations=1)
cv2.imwrite('dilation.jpg', dilation)