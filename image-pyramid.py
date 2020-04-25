import cv2

image = cv2.imread('image.jpg')
lower_reso = cv2.pyrDown(image)
cv2.imwrite('low_res.jpg', lower_reso)
