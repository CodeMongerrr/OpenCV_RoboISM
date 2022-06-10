import cv2
import numpy as np
import matplotlib.pyplot as plt
contours = 0
def ImageProcess(img):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img = cv2.imread('CVtask.jpg')
ImageProcess(img)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        asp1 = float(x1)/w
        asp2 = float(y1)/h
        #print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            if a == 0:
                pass
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(img, (x1, y1) , (x1 + w, y1 + w), (0, 255, 0), 10)
            x2 = x1+w
            y2 = y1+h