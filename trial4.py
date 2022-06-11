import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import math
contours = 0



img = cv2.imread('HaHa.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
corners = np.array(corners)
corners.resize(4,2)
print(corners)
print()
img1 = imutils.rotate(img,180/math.pi*math.atan((int(corners[1][1]) - int(corners[0][1]))/(int(corners[1][0]) - int(corners[0][0]))), scale= 0.8)
(corners, ids, rejected) = cv2.aruco.detectMarkers(img1, arucoDict, parameters=arucoParams)
corners = np.array(corners)
corners.resize(4,2)
print(corners)
cv2.imshow("sd", img1)



#for i in range(len(corners)):
#    plt.scatter(corners[i][0], corners[i][1])
#plt.show()
#img = np.array(img)
#img1 = img[78:515, 77:515]
#cv2.imshow("asd", img1)
"""
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
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(img, (x1, y1) , (x1 + w, y1 + w), (0, 255, 0), 10)
            x2 = x1+w
            y2 = y1+h
            """
cv2.waitKey(0)
cv2.destroyAllWindows()