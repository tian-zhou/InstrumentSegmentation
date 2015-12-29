"""
-------------------------------------------------------------------------------
Name:        test module
Purpose:     test purpose
Idea:        how to solve it
Author:      Tian Zhou
Email:       zhou338 [at] purdue [dot] edu
Created:     19/11/2015
Copyright:   (c) Tian Zhou 2015
-------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def GradientAnalysis(imgGray):
    plt.figure()
    laplacian = cv2.Laplacian(imgGray,cv2.CV_64F)
    cv2.normalize(laplacian, laplacianNormed, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sobelx = cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=5)

    plt.subplot(2,2,1),plt.imshow(imgGray,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    HistAnalysis(laplacianNormed)

def HistAnalysis(imgGray, imgMask = None):
    plt.figure()
    hist_full = cv2.calcHist([imgGray],[0],None,[256],[0,256])
    if imgMask == None:
        plt.plot(hist_full)
        plt.xlim([0,256])
    else:
        cv2.imshow('imgMask', imgMask)
        cv2.waitKey(1)
        plt.subplot(121)
        plt.plot(hist_full)
        plt.xlim([0,256])
        hist_mask = cv2.calcHist([imgGray],[0],imgMask,[256],[0,256])
        plt.subplot(122)
        plt.plot(hist_mask)
        plt.xlim([0,256])

def main():
    GradientAnalysis(imgGray)
    HistAnalysis(imgGray)
    _,imgMask = cv2.threshold(imgMask,127,255,cv2.THRESH_BINARY)
    GradientAnalysis(imgDepth)
    # HistAnalysis(imgDepth, imgMask)

    plt.show()
    print 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
