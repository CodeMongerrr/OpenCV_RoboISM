import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('CVtask.jpg')
img_arr = np.array(img)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
m = np.array(contours)
print(contours)

for i in range(1240):
    for j in range(1724):
           if img_arr[i][j][0] >=230 and img_arr[i][j][1] >=230 and img_arr[i][j][2] >= 230:
               img_arr[i][j][0] = 0
               img_arr[i][j][1] = 0
               img_arr[i][j][2] = 0

"""            
for i in range(1240):
    for j in range(1724):
        if img_arr[i][j][0] > 140 and img_arr[i][j][0] < 150 and img_arr[i][j][1] > 200 and img_arr[i][j][1] < 210 and img_arr[i][j][2] < 85 and img_arr[i][j][2] > 75:
            a = 1
        else:
            img_arr[i][j][0] = 0
            img_arr[i][j][1] = 0
            img_arr[i][j][2] = 0
"""
cv2.imshow("Hi", img_arr)
plt.imshow(img)
#ne = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#cv2.imshow("nnene", ne)










#cv2.imshow("cropped image", a)



cv2.waitKey(0)
cv2.destroyAllWindows()