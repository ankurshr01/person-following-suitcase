import numpy as np
import cv2  

cap = cv2.VideoCapture(0)

hp = 0
xp = 0
yp = 0
while True:
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange (hsv, lower_blue, upper_blue)
    bluecnts = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)
        sum_x=0
        sum_y=0
        sum_w=0
        sum_h=0
        for x in range(100):
            sum_x=sum_x+xg
            sum_y=sum_y+yg
            sum_w=sum_w+wg
            sum_h=sum_h+hg
            if(x==99):
                if sum_x<xp:
                    print("right: ", str(xp-sum_x)+" px/frame |",end='')
                elif sum_x>xp:
                    print(" left: ", str(sum_x-xp)+" px/frame |",end='')
                else:
                    print(' same: ', str(xp-sum_x)+" px/frame |",end='')
                xp = sum_x
                if sum_y>yp:
                    print("down: ", str(sum_y-yp)+" px/frame |",end='')
                elif sum_y<yp:
                    print(" top: ", str(yp-sum_y)+" px/frame |",end='')
                else:
                    print('same: ', str(sum_y-yp)+" px/frame |",end='')
                yp = sum_y


                if sum_h<hp:
                    print('far', str(hp-sum_h)+" px/frame |",end='')
                elif sum_h>hp:
                    print('near', str(sum_h-hp)+" px/frame |",end='')
                else:
                    print('same', str(sum_h-hp)+" px/frame |",end='')
                hp = sum_h
                i=0;
        
        


    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    k = cv2.waitKey(5) 
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
