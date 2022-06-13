import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import math

CV = cv2.imread('CVtask.jpg')
imgGrey = cv2.cvtColor(CV, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", CV)
sqs = []
a = 0
Arucos = [0,0,0,0]

Arucos[0] = cv2.imread('Ha.jpg')
Arucos[1] = cv2.imread('HaHa.jpg')
Arucos[2] = cv2.imread('XD.jpg')
Arucos[3] = cv2.imread('LMAO.jpg')
val = []
col = []
prob = {"green":1, "orange":2, "black":3, "pink":4}
colour = ""
corrected = [0, 0, 0, 0]


# def overlay(image1, image2, x, y):
#     image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     contours, _ = cv2.findContours(image1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     image1_mask = np.zeros_like(image1)
#     cv2.drawContours(image1_mask, contours, -1, (255,255,255), -1)
#     idx = np.where(image1_mask == 255)
#     image2[y+idx[0], x+idx[1], idx[2]] = image1[idx[0], idx[1], idx[2]]
#     return image2


def FindArucoMarkers(img):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    arucoParam = cv2.aruco.DetectorParameters_create()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    corners = np.array(corners)
    corners.resize(4, 2)
    #print(corners)
    img1 = imutils.rotate(img, 180 / math.pi * math.atan((int(corners[1][1]) - int(corners[0][1])) / (int(corners[1][0]) - int(corners[0][0]))), scale=0.8)
    #cv2.imshow(f"{ids[0][0]}", img1)
    corners, ids, rejected = cv2.aruco.detectMarkers(img1, arucoDict, parameters=arucoParam)
    corners = np.array(corners)
    corners.resize(4, 2)
    #print(corners)
    img1 = img1[int(corners[0][0]):int(corners[2][0]), int(corners[0][1]): int(corners[2][1])]
    center_aruco = [img1.shape[0]//2, img.shape[1]//2]
    corrected[ids[0][0] - 1] = np.array(img1)
    cv2.imshow(f"{ids[0][0]}", img1)
    return np.array(img1)


for i in range(4):
    val.append(FindArucoMarkers(Arucos[i]))
    FindArucoMarkers(Arucos[i])

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(CV, [approx], 0, (0, 0, 0), 1)

    #plt.imshow("assas", CV)
    #plt.scatter(x,y)
    plt.show()
    if len(approx) == 4:

        x1 ,y1, w, h = cv2.boundingRect(approx)
        Y = y1
        X = x1
        aspectRatio = float(w) / h
        #print(X, Y)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            if a == 0:
                pass
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            print(x, y)
            #imsdas = overlay(Arucos[0], CV, x1, y1)
            #cv2.imshow("ADad", imsdas)
            cv2.putText(CV, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(CV, (x1, y1) , (x1 + w, y1 + w), (0, 255, 0), 1)
            x2 = x1+w
            y2 = y1+h
            color = CV[int(y1+h/2)][int(x1+w/2)]
            #if color[0] < 20 and color[1] > 240 and color[2] <20:
                 # x, y, w, h = cv2.boundingRect(contour)
                 # Y = y
                 # X = x
                 # h1 , w1 = Arucos[0].shape[0], Arucos[0].shape[1]
                 # h2, w2 = x1, y1
                 # print(Y, X)
                 # if h2 - Y > h1 + 1 and w2 - X > w1 + 1:
                 #     CV[X:X + h1, Y:Y + w1] = val[2]


            cv2.imshow("Contours", CV)

#imsdas = overlay(Arucos[0], CV, 0, 0)
cv2.imshow("shapes", CV)
cv2.waitKey(0)
cv2.destroyAllWindows()