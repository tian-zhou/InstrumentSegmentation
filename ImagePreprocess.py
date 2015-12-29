"""
-------------------------------------------------------------------------------
Name:        test module
Purpose:     test purpose
Idea:        how to solve it
Time:        N/A
Space:       N/A
Author:      Tian Zhou
Email:       zhou338 [at] purdue [dot] edu
Created:     01/12/2015
Copyright:   (c) Tian Zhou 2015
-------------------------------------------------------------------------------
"""

import numpy as np
import cv2

"""
utility for the training images.
1) rename the images
2) resize the depth and mask using color size
3) mask the color and depth image using the mask

"""
def main():

    # read in image
    colorFileName = 'training\\color_'+str(fileIndex)+'.jpg'
    depthFileName = 'training\\depth_'+str(fileIndex)+'.jpg'
    maskFileName = 'training\\mask_'+str(fileIndex)+'.jpg'
    imgBGR = cv2.imread(colorFileName, 1)
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    imgDepth = cv2.imread(depthFileName, 0)
    imgMaskGray = cv2.imread(maskFileName, 0)
    # erod the mask so that the boundary won't bother us
    imgMaskGray = cv2.erode(imgMaskGray, np.ones((5,5),np.uint8))
    _, imgMask = cv2.threshold(imgMaskGray,127,255,cv2.THRESH_BINARY)

    # rescale depth and mask
    imgDepth, imgMask = self.RescaleDepth(imgBGR, imgDepth, imgMask)

    # histogram equalization on the gray image
    imgGrayHistEqu = self.IlluminationNorm(imgGray)

    if show == True:
        cv2.imshow("imgBGR", imgBGR)
        cv2.imshow("imgGray", imgGray)
        cv2.imshow('imgDepth', imgDepth)
        cv2.imshow('imgMask', imgMask)
        WaitKey(1)
    return imgBGR, imgGray, imgGrayHistEqu, imgDepth, imgMask

if __name__ == '__main__':
    main()
