import cv2
import numpy as np

image = cv2.imread('image.jpg', 0)
kernel = np.ones((5, 5), np.uint8)

"""
opening: erosion followed by dilation
closing: dilation followed by erosion 

"""

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('opening.jpg', opening)
cv2.imwrite('closing.jpg', closing)