#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Tian
#
# Created:     24/11/2015
# Copyright:   (c) Tian 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import numpy as np

def printBar():
    print ''.join(['-' for i in range(40)])

def printStar():
    print ''.join(['*' for i in range(40)])

def WaitKey(time):
    #key = cv2.waitKey(1)
    key = cv2.waitKey(time)
    if key == 27:
        cv2.destroyAllWindows()
        exit()

def main():
    printBar()
    printStar()

if __name__ == '__main__':
    main()
