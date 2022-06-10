import cv2
import numpy as mp
import numpy as np

img = cv2.imread('CVtask.jpg')
img1 = cv2.imread('Ha.jpg')
arr = np.array(img)
arr1 =np.array(img1)
i1 = 0
jk = 0
for i in range(arr.shape[0]):

    for j in range(arr.shape[1]):

        if i >200 and i<= 200 + arr1.shape[0]+1 and j > 200 and j <= 200 + arr1.shape[1]+1:
            arr[i][j][0] = arr1[i1][jk][0]
            arr[i][j][1] = arr1[i1][jk][1]
            arr[i][j][2] = arr1[i1][jk][2]
            jk += 1
    if i > 200 and i < 200 + arr1.shape[0] and j > 200 and j < 200 + arr1.shape[1]:
        i1 += 1
        print(i1)
cv2.imshow("img", arr)