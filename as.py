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
        Y = y
        X = x
        print(Y, X)

        if h2 - Y > h1 + 1 and w2 - X > w1 + 1:
                bg_img[Y+500:Y + h1+500, X:X + w1] = fg_img

        cv2.imshow("Contours", bg_img)

cv2.waitKey(0)