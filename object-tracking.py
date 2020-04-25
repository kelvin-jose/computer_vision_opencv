import cv2
import numpy as np

cap = cv2.VideoCapture('/home/kelvin/Accubits/garbage/cv/garbage/un_processed/1_222.mp4')

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 255, 255])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5)
    if k == 27:
        break

cv2.destroyAllWindows()

