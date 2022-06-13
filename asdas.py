import
CV = cv2.imread('CVtask.jpg')
imgGrey = cv2.cvtColor(CV, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", CV)



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
            cv2.putText(CV, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(CV, (x1, y1) , (x1 + w, y1 + w), (0, 255, 0), 1)

