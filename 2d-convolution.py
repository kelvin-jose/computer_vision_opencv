import cv2
import numpy as np

image = cv2.imread('image.jpg')
kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(image, -1, kernel)
cv2.imwrite('2d-filter.jpg', dst)

