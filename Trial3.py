import matplotlib.pyplot as plt
import numpy as np
import cv2

#importing the image
img = cv2.imread('CVtask.jpg')

#Turning it to gray
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#assigning the threshold for the image
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)

#grabbing  the contours in the image
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)


    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    #print(approx)
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


    elif len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        asp1 = float(x1)/w
        asp2 = float(y1)/h
        #print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))



        elif asp1 >=0.95 and asp2 <= 1.05:
            cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        else:
            cv2.putText(img, "trapezium", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))