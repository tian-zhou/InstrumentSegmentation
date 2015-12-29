import cv2
import numpy as np
from matplotlib import pyplot as plt

drawing = False # true is mouse pressed
ix, iy = -1,-1
button = 0

# mouse callback function
def draw_rect(event,x,y,flags,param):
    global ix,iy,drawing,mode, button, GreenMask, RedMask

    if event == cv2.EVENT_LBUTTONDOWN:
        button = 0
        drawing = True
        ix,iy = x,y # record top left coordinate of rect

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

##    elif event == cv2.EVENT_RBUTTONDOWN:
##        button = 1
##        drawing = True
##        ix, iy = x,y # record top left coordinate of rect

    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if button == 0:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                cv2.rectangle(GreenMask, (ix,iy), (x,y), 255,-1)
                RedMask = 255 - GreenMask
##            else:
##                cv2.rectangle(img,(ix,iy), (x,y), (0,0,255),-1)
##                cv2.rectangle(RedMask, (ix,iy), (x,y), 255,-1)


img = cv2.imread('ColorMayo.jpg')
imgCopy = img.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rect)


GreenMask = np.zeros(img.shape[0:2], np.uint8)
RedMask = 255 * np.ones(img.shape[0:2], np.uint8)

"""
The flag to decide if we want to load the existing selection, so that we don't
need to do it every time
"""
loadSelectionFlag = 0

print "loadSelectionFlag: ", loadSelectionFlag
print "Press d when you are finished selecting..."
while(True):
    cv2.imshow('image', img)
    cv2.imshow('GreenMask', GreenMask)
    cv2.imshow('RedMask', RedMask)
    k = cv2.waitKey(1)
    if k == ord('d'):
        if loadSelectionFlag == 1:
            GreenImg = cv2.imread('GreenImg.jpg')
            RedImg = cv2.imread('RedImg.jpg')
            GreenMask = cv2.imread('GreenMask.jpg')
            GreenMask = cv2.cvtColor(GreenMask, cv2.COLOR_BGR2GRAY)
            _, GreenMask = cv2.threshold(GreenMask,128,255,0)
            RedMask = cv2.imread('RedMask.jpg')
            RedMask = cv2.cvtColor(RedMask, cv2.COLOR_BGR2GRAY)
            _, RedMask = cv2.threshold(RedMask,128,255,0)
        else:
            GreenImg = cv2.bitwise_and(imgCopy,imgCopy,mask = GreenMask)
            RedImg = cv2.bitwise_and(imgCopy,imgCopy,mask = RedMask)
            cv2.imwrite("GreenMask.jpg", GreenMask)
            cv2.imwrite("RedMask.jpg", RedMask)
            cv2.imwrite("GreenImg.jpg", GreenImg)
            cv2.imwrite("RedImg.jpg", RedImg)

        cv2.imshow("GreenImg", GreenImg)
        cv2.imshow("RedImg", RedImg)
        cv2.imshow('GreenMask', GreenMask)
        cv2.imshow("RedMask", RedMask)
        cv2.waitKey(0)

        imgHSV = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)

        GreenImgHSV = cv2.cvtColor(GreenImg, cv2.COLOR_BGR2HSV)
        GreenImgHist = cv2.calcHist([GreenImgHSV], [0,1], GreenMask, [180,256], [0,180,0,256])
        cv2.normalize(GreenImgHist, GreenImgHist, 0, 255, cv2.NORM_MINMAX)
        GreenProb = cv2.calcBackProject([imgHSV], [0,1], GreenImgHist,[0,180,0,256],1)
        #cv2.imshow("GreenProb", GreenProb)

        RedImgHSV = cv2.cvtColor(RedImg, cv2.COLOR_BGR2HSV)
        RedImgHist = cv2.calcHist([RedImgHSV], [0,1], RedMask, [180,256], [0,180,0,256])
        cv2.normalize(RedImgHist, RedImgHist, 0, 255, cv2.NORM_MINMAX)
        RedProb = cv2.calcBackProject([imgHSV], [0,1], RedImgHist,[0,180,0,256],1)
        #cv2.imshow("RedProb", RedProb)

        # Now convolute with circular disc
        #disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #cv2.filter2D(GreenProb,-1,disc,GreenProb)
        #cv2.filter2D(RedProb,-1,disc,RedProb)

        #GreenProbMedian = cv2.medianBlur(GreenProb, 5)
        #RedProbMedian = cv2.medianBlur(RedProb, 5)
        #cv2.imshow("GreenProbMedian", GreenProbMedian)
        #cv2.imshow("RedProbMedian", RedProbMedian)

        #ret,Greenthresh = cv2.threshold(GreenProbMedian,50,255,0)
        #ret,Redthresh = cv2.threshold(RedProbMedian,50,255,0)
        #cv2.imshow('Greenthresh', Greenthresh)
        #cv2.imshow('Redthresh', Redthresh)
        #cv2.waitKey(0)


        # np.divide(1,0)
        # threshold and binary AND
        ratio = 4 # 1 is too low, can empiracally choose
        RedProb[RedProb==0] = 1 #np.divide(255,0) = 0, not desired. Change 0 to 1
        FGMask = np.array(255 * (np.divide(GreenProb, RedProb) > ratio), np.uint8)
        FGMaskMedian = cv2.medianBlur(FGMask, 9)
        FGMaskMedian = cv2.merge((FGMaskMedian,FGMaskMedian,FGMaskMedian))
        cv2.imshow('FGMaskMedian', FGMaskMedian)
        Foreground = cv2.bitwise_and(imgCopy,FGMaskMedian)
        cv2.imshow('Foreground', Foreground)
        cv2.imwrite("FGMaskMedian.jpg", FGMaskMedian)

        cv2.waitKey(0)
        break

    elif k == 27:
        break

cv2.destroyAllWindows()
print "python script finished"
