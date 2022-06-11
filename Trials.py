import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import math

img = cv2.imread('CVtask.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", img)
sqs = []
a = 0
Arucos = [0,0,0,0]
Arucos[0] = cv2.imread('Ha.jpg')
Arucos[1] = cv2.imread('HaHa.jpg')
Arucos[2] = cv2.imread('XD.jpg')
Arucos[3] = cv2.imread('LMAO.jpg')
val = []
col = []
corrected = {1: 0, 2: 0, 3: 0, 4: 0}
def FindArucoMarkers(img):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    corners = np.array(corners)
    corners.resize(4, 2)
    print(corners)
    img1 = imutils.rotate(img, 180 / math.pi * math.atan((int(corners[1][1]) - int(corners[0][1])) / (int(corners[1][0]) - int(corners[0][0]))), scale=0.8)
    corners, ids, rejected = cv2.aruco.detectMarkers(img1, arucoDict, parameters=arucoParam)
    corners = np.array(corners)
    corners.resize(4, 2)
    print(corners)
    img1 = img1[int(corners[0][0]):int(corners[2][0]), int(corners[0][1]): int(corners[2][1])]
    corrected[id[0][0]] = img1
    cv2.imshow(f"{ids[0][0]}", img1)


for i in range(4):
    FindArucoMarkers(Arucos[i])

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
            color = img[int(y1+h/2), int(x1+w/2)]
            col.append(color)
            rot = cv2.getRotationMatrix2D((int(y1+h/2), int(x1+w/2)), 50, 1)





plt.imshow(img)
plt.show()
cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()