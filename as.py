import imutils
import cv2

fg_img = cv2.imread("Ha.jpg")
bg_img = cv2.imread("CVtask.jpg")

graybg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

h1, w1 = fg_img.shape[:2]
print(h1, w1)

thresh = cv2.threshold(graybg, 225, 255, cv2.THRESH_BINARY)[1]
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=1)
mask2 = mask.copy()
mask2 = cv2.dilate(mask2, None, iterations = 2)
h2, w2 = mask2.shape[:2]
print(h2, w2)

cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        pip_h = y
        pip_w = x
        print(pip_h, pip_w)

        if h2 - pip_h > h1 + 1 and w2 - pip_w > w1 + 1:
                bg_img[pip_h:pip_h+h1,pip_w:pip_w+w1] = fg_img

        cv2.imshow("Contours", bg_img)

cv2.waitKey(0)