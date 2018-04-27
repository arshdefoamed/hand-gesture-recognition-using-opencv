# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:08:01 2018

@author: arshd
"""
import numpy as np
import cv2
import math
import pyautogui
from threading import Timer

# Open Camera
capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
ch=True
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def calcAngle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.linalg.norm(v1)
    y_modulus = np.linalg.norm(v2)
    cos_angle = (dot / x_modulus )/ y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def get_palm_circle(contourPts, origImg):
    dist_max = np.zeros((origImg.shape[0], origImg.shape[1]))
    for y in range(0, origImg.shape[0], 4):
        for x in range(0, origImg.shape[1], 4):
            if origImg[y, x]:
                dist_max[y, x] = cv2.pointPolygonTest(contourPts, (x, y), True)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_max)
    return max_loc, max_val

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine
def getMaxContour(contoursVec):
#    maximumCtrArea=700
#    thresholdedCtrs=[]
    max_area=0
    maxContour=[]
    for i in range(len(contoursVec)):
        extractedContour=contoursVec[i]
        ctrArea = cv2.contourArea(extractedContour)
        if(ctrArea>max_area):
            max_area=ctrArea
#            thresholdedCtrs.append(extractedContour)
            maxContour=extractedContour
    return maxContour
                
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
kernel_square = np.ones((11,11),np.uint8)
kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_ellipse2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))



fireLeftFlag=True
fireRightFlag=True
fireUpFlag=True
fireDownFlag=True
def turnFlagToTrue(ch):
    print(ch)
    global fireLeftFlag
    global fireRightFlag
    global fireUpFlag
    global fireDownFlag
    if(ch=="L"):
        fireLeftFlag=True;
    elif(ch=="R"):
        fireRightFlag=True;
    elif(ch=="U"):
        fireUpFlag=True;
    elif(ch=="D"):
        fireDownFlag=True;
try:
    while capture.isOpened():
#        get camera input
        ret, frame = capture.read()
        frame = np.fliplr(frame)
        blur = cv2.blur(frame,(3,3))
        
#        skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skinMask  = cv2.inRange(hsv, np.array([54,131,110]), np.array([163,157,135]))
        
#        perform morphology to remove holes
        morphedFrame=skinMask
        morphedFrame = cv2.dilate(morphedFrame,kernel_ellipse,iterations = 1)
        morphedFrame = cv2.erode(morphedFrame,kernel_square,iterations = 1)    
        morphedFrame = cv2.dilate(morphedFrame,kernel_ellipse,iterations = 1)    
        morphedFrame = cv2.medianBlur(morphedFrame,5)
        morphedFrame = cv2.erode(morphedFrame,kernel_ellipse,iterations = 1)    
        morphedFrame = cv2.dilate(morphedFrame,kernel_ellipse2,iterations = 1)
        morphedFrame = cv2.dilate(morphedFrame,kernel_ellipse,iterations = 1)
        morphedFrame = cv2.medianBlur(morphedFrame,5)
        
#        find contours
        contouredPic,contoursVec, hierarchy = cv2.findContours(morphedFrame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        maxContour=getMaxContour(contoursVec);
        origContouredFrame=frame.copy()
        cv2.drawContours(origContouredFrame, maxContour, -1, (0,122,122), 3)
        hull = cv2.convexHull(maxContour)
        
        topMostPoint=[]
        for ptr in range(hull.shape[0]):
            if(np.size(topMostPoint)==0 or (hull[ptr,0,1] < topMostPoint[1])):
                topMostPoint=hull[ptr,0]
#        if (np.size(topMostPoint)>0 ) :
#            cv2.putText(origContouredFrame,str(topMostPoint),(topMostPoint[0],topMostPoint[1]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
        
        gestureThresh=30
# swipe detection
        timeOut=2
#        cv2.line(origContouredFrame,(gestureThresh,1),(gestureThresh,300),[255,255,0],2)
#        cv2.line(origContouredFrame,(np.shape(frame)[1]-gestureThresh,1),(np.shape(frame)[1]-gestureThresh,300),[255,255,0],2)

        if(topMostPoint[0]<gestureThresh):
            cv2.putText(origContouredFrame,"fire left",(23,59),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
            if(fireLeftFlag):
                fireLeftFlag= False
                pyautogui.press('left')
                t= Timer(timeOut,turnFlagToTrue,["L"])
                t.start()
        elif(topMostPoint[0]>np.shape(frame)[1]-gestureThresh):
            cv2.putText(origContouredFrame,"fire right",(23,59),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
            if(fireRightFlag):
                fireRightFlag= False
                pyautogui.press('right')
                t= Timer(timeOut,turnFlagToTrue,["R"])
                t.start()
# swipe up and down is not working really well now.
                
#        elif(topMostPoint[1]<gestureThresh):
#            cv2.putText(origContouredFrame,"fire up",(23,59),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
#            if(fireUpFlag):
#                pyautogui.press('up')
#                t= Timer(timeOut,turnFlagToTrue,["U"])
#                t.start()
#        elif(topMostPoint[1]>np.shape(frame)[0]-gestureThresh):
#            cv2.putText(origContouredFrame,"fire down",(23,59),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
#            if(fireDownFlag):
#                pyautogui.press('down')
#                t= Timer(timeOut,turnFlagToTrue,["D"])
#                t.start()

# swipe detection ends
        cv2.drawContours(origContouredFrame,[hull],-1,(255,255,255),2)
        x,y,w,h = cv2.boundingRect(maxContour)
        centerCoord, circRadius = get_palm_circle(maxContour, morphedFrame)
        cv2.circle(origContouredFrame,(centerCoord[0],centerCoord[1]),int(circRadius),(0, 255, 0),1)
        origContouredFrame = cv2.rectangle(origContouredFrame,(centerCoord[0],centerCoord[1]),(centerCoord[0],centerCoord[1]),(125,255,0),12)
        
        cHull2 = cv2.convexHull(maxContour,returnPoints = False)
        defects = cv2.convexityDefects(maxContour,cHull2)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(maxContour[s][0])
            end = tuple(maxContour[e][0])
            far = tuple(maxContour[f][0])
            cv2.circle(origContouredFrame,far,5,[0,0,255],-1)
        
        
#        limiting region of interest
        scaleFactorForROI=3.5
        roiTopLeftRowNum=round(centerCoord[1]-scaleFactorForROI*circRadius)
        roiTopLeftColNum=round(centerCoord[0]-scaleFactorForROI*circRadius)
        if(roiTopLeftRowNum<0):
            roiTopLeftRowNum=0
        if(roiTopLeftColNum<0):
            roiTopLeftColNum=0
        roiBottomRightRowNum=round(centerCoord[1]+(1.5*circRadius))
        roiBottomRightColNum=round(centerCoord[0]+(scaleFactorForROI*circRadius))
    
        if(roiBottomRightRowNum>np.shape(frame)[0]):
            roiBottomRightRowNum=np.shape(frame)[0]
        if(roiBottomRightColNum>np.shape(frame)[1]):
            roiBottomRightColNum=np.shape(frame)[1]
        origContouredFrame = cv2.rectangle(origContouredFrame,(int(roiTopLeftColNum),int(roiTopLeftRowNum)),(int(roiBottomRightColNum),int(roiBottomRightRowNum)),(0,255,0),2)
        
        
        roiFrame = morphedFrame[roiTopLeftRowNum:roiBottomRightRowNum,roiTopLeftColNum:roiBottomRightColNum].copy();
        roiFrameColored = frame[roiTopLeftRowNum:roiBottomRightRowNum,roiTopLeftColNum:roiBottomRightColNum,:].copy();
# contour extraction on ROI
        contouredPic,contoursVec, hierarchy = cv2.findContours(roiFrame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        maxContour=getMaxContour(contoursVec);
# finding max inscribed circle
        maxInscrbdCirclCenterCoord, maxInscrbdCirclRadius = get_palm_circle(maxContour, roiFrame)
        maxInscrbdCirclCenterCoord=(int(maxInscrbdCirclCenterCoord[0]),int(maxInscrbdCirclCenterCoord[1]))
        cv2.circle(roiFrameColored,(int(maxInscrbdCirclCenterCoord[0]),int(maxInscrbdCirclCenterCoord[1])),int(maxInscrbdCirclRadius),(0, 255, 0),1)
        cv2.circle(roiFrameColored,maxInscrbdCirclCenterCoord,5,[0,255,0],-1)
# finding min enclosed circle
        minEnclCircleCenter,minEnclCircleCenterRadius = cv2.minEnclosingCircle(maxContour)
        minEnclCircleCenter=(int(minEnclCircleCenter[0]),int(minEnclCircleCenter[1]))
        cv2.circle(roiFrameColored,(minEnclCircleCenter[0],minEnclCircleCenter[1]),int(minEnclCircleCenterRadius),(255, 255, 0),1)
        cv2.circle(roiFrameColored,minEnclCircleCenter,5,[255,255,255],-1)

        cHull = cv2.convexHull(maxContour)
        cv2.drawContours(roiFrameColored,[cHull],-1,(150,0,0),2)
        cHull2 = cv2.convexHull(maxContour,returnPoints = False)
        defects = cv2.convexityDefects(maxContour,cHull2)
        convexPointsArr=[]
        ctr=1
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(maxContour[s][0])
            end = tuple(maxContour[e][0])
            far = tuple(maxContour[f][0])
           
            dist = DistancePointLine(far[0],far[1],start[0],start[1],end[0],end[1])
            if (dist>maxInscrbdCirclRadius and dist<minEnclCircleCenterRadius):
                vectSF = [start[0]-far[0],start[1]-far[1]]
                vectEF = [end[0]-far[0],end[1]-far[1]]
                angleBwVecs=calcAngle(vectSF,vectEF)
                if(angleBwVecs<90):
                    convexPointsArr.append([start,far,end])
                    cv2.line(roiFrameColored,start,end,[0,255,0],2)
                    cv2.line(roiFrameColored,start,far,[0,255,255],2)
                    cv2.line(roiFrameColored,far,end,[255,255,0],2)
                    cv2.circle(roiFrameColored,far,5,[0,0,255],-1)
                    ctr=ctr+1

        thresh =30
        #finger count calculation
        if (ctr==1 and abs(minEnclCircleCenter[0]-maxInscrbdCirclCenterCoord[0])+ abs(minEnclCircleCenter[1]-maxInscrbdCirclCenterCoord[1])<thresh):
            cv2.putText(origContouredFrame,"Zero",(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
        else:
            cv2.putText(origContouredFrame,str(ctr),(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)

        cv2.imshow("orig", frame )
        cv2.imshow("Thresholded", skinMask )
        cv2.imshow("morphedFrame", morphedFrame )
        cv2.imshow("roiFrameColored", roiFrameColored )
        cv2.resizeWindow("roiFrameColored", 400,400 )
        cv2.imshow("origContouredFrame", origContouredFrame )
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()
except Exception as e:
    print('there was an error')
    print(e)
    capture.release()
    cv2.destroyAllWindows()
