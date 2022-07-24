import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
import math


img = cv2.imread("CVtask.jpg")
A1 = (cv2.imread("Ha.jpg"))
A2 = (cv2.imread("HaHa.jpg"))
A3 = (cv2.imread("LMAO.jpg"))
A4 = (cv2.imread("XD.jpg"))

A_List = (A1, A2, A3, A4)


def findAruco(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(gray,arucoDict,parameters = arucoParam)
    return (corners,ids,rejected)


id_list = []
for i in A_List:
    id_list.append((findAruco(i))[1][0][0])
#print(id_list)


def aruco_coordinates(img,aru_length,aru_height,bound_height,bound_length,angle):
    (c,i,r) = findAruco(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (tl,tr,br,bl) = corner
            bl = (int(bl[0]),int(bl[1]))
            br = (int(br[0]),int(br[1]))
            m = ((br[1]-bl[1])/(br[0]-bl[0]))           #finding slope
            fi = math.atan(m)
            a = fi * 180/math.pi                             #finding inclined angle of arucos
            # print(np.shape(img))
            img = imutils.rotate_bound(img,-a)               #rotating the arucos
            (c, i, r) = findAruco(img)
            if len(c) > 0:
                i = i.flatten()
                for (markercorner, markerid) in zip(c, i):
                    corner = markercorner.reshape((4, 2))
                    (tl, tr, br, bl) = corner


                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    img = img[tl[1]:br[1],tl[0]:br[0]]
            blank=np.zeros((int(bound_height),int(bound_length),3))
            s=np.shape(blank[ int( (int(bound_height) - int(aru_height)) / 2) : int((int(bound_length) + int(aru_length)) / 2), int((int(bound_length) - int(aru_length)) / 2) : int((int(bound_length) + int(aru_length)) / 2)])
            #print(s)
            img=cv2.resize(img,(s[1],s[0]))       #resizing the image
            blank[int((int(bound_height)-int(aru_height))/2):int((int(bound_length)+int(aru_length))/2),int((int(bound_length)-int(aru_length))/2):int((int(bound_length)+int(aru_length))/2)]=img
            img=imutils.rotate(blank,angle)       #rotating the blank
    return img


def color(color,lower,upper):     #function to detect color of squares
    if color[0] in range(lower[0],upper[0]+1):
        if color[1] in range(lower[1],upper[1]+1):
            if color[2] in range(lower[2], upper[2] + 1):
                return True
    else:
        return False


def Result(img,id_list):    #function for detecting squares,its contours and imposing the aruco on the respective squares
        img = cv2.imread("CVtask.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret , thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        cont,heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in cont:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 4:
                if float(w)/h >= 0.95 and float(w)/h <= 1.05:

                    slope = (approx[1][0][1]-approx[2][0][1])/(approx[1][0][0]-approx[2][0][0])
                    angle = 180/math.pi*math.atan(slope)
                    dx = math.sqrt((approx[1][0][1]-approx[2][0][1])**2 + (approx[1][0][0]-approx[2][0][0])**2)
                    dx = int(dx)
                    dy = math.sqrt((approx[2][0][1] - approx[3][0][1]) ** 2 + (approx[2][0][0] - approx[3][0][0]) ** 2)
                    dy = int(dy)

                    if color(img[int(y+(h/2)) , int(x+(w/2))],(0,120,0),(150,255,150)): #green
                        ind = id_list.index(1)
                        aruco_img = aruco_coordinates(A_List[ind],dx,dy,h,w,-angle)
                        print(1)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(0,100,200),(150,200,255)): #orange
                        ind = id_list.index(2)
                        aruco_img = aruco_coordinates(A_List[ind],dx,dy,h,w,-angle)
                        print(2)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(0,0,0),(20,20,20)): #black
                        ind = id_list.index(3)
                        aruco_img = aruco_coordinates(A_List[ind],dx,dy,h,w,-angle)
                        print(3)
                    elif color(img[int(y+(h/2)),int(x+(w/2))],(200,200,200),(250,250,250)): #pink-peach
                        ind = id_list.index(4)
                        aruco_img = aruco_coordinates(A_List[ind],dx,dy,h,w,-angle)
                        print(4)
                    cv2.drawContours(img,[approx],-1,(0,0,0),-1)
                    img[y:y+h,x:x+w] = img[y:y+h,x:x+w]+ aruco_img
        return img

Result(img,id_list)

cv2.namedWindow("FINAL_IMAGE",cv2.WINDOW_NORMAL)
cv2.imshow("FINAL_IMAGE",Result(img,id_list))
cv2.waitKey(0)
cv2.destroyAlowerWindows()