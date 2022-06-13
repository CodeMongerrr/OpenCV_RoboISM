import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('CVtask.jpg')
img_arr = np.array(img)
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
m = np.array(contours[0])
print(contours)
cv2.waitKey(0)
cv2.destroyAllWindows()