import cv2

image = cv2.imread('image.jpg', 0)
edge = cv2.Canny(image, 200, 400)
cv2.imwrite('canny.jpg', image)