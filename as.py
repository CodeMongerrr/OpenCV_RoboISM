import cv2
import imutils
import numpy as mp
import numpy as np

img = cv2.imread('CVtask.jpg')
img1 = cv2.imread('Ha.jpg')
arr = np.array(img)
arr1 =np.array(img1)
i1 = 0
jk = 0
for i in range(img1.shape[0]):
    for i in range(img1.shape[1]):
        if img1[i][j][0] == 255 and img1[i][j][1] == 255 and img1[i][j][2] == 255:
            img1

a = imutils.rotate(img1, 30, scale= 0.6)
cv2.imshow("asa", a)

cv2.waitKey(0)
cv2.destroyAllWindows()