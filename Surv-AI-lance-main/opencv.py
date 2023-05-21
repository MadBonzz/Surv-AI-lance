import cv2
import numpy as np
vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('g'):
        break

vid.release()

cv2.destroyAllWindows()
