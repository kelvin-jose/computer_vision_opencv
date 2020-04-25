import cv2
import numpy as np

image = cv2.imread('image.jpg', 0)

cv_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
"""
Arguments of cv2.calcHist
    1. [image]: image
    2. [0]: Channels, we're using a grayscale image, so the value is 0
    3. None: It denotes whether or not the image should be masked. if 
             we want to find histogram of the complete image None is 
             enough, But if you want to find histogram of particular 
             region of image, you have to create a mask image for that
             and give it as mask.
    4. [256]: Number of bins
    5. [0, 256]: Pixel range 
"""

np_hist, bins = np.histogram(image.ravel(), 256, [0, 256])
"""
Arguments of numpy.histogram
    1. image.ravel(): image
    2. 256: Number of bins
    3. [0, 256]: Pixel range
"""
