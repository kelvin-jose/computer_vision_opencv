import cv2

image = cv2.imread('image.jpg')
blurred = cv2.blur(image, (5, 5))  # 5x5 kernel
cv2.imwrite('blurred.jpg', blurred)
