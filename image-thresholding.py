import cv2

img = cv2.imread('image.jpg', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [thresh1, thresh2, thresh3, thresh4, thresh5]

for image, title in zip(images, titles):
    cv2.imwrite(title + '.jpg', image)


