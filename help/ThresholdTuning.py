import cv2
import numpy as np

def test():
    'read in the image'
    imgBGR = cv2.imread('training\\depth_1.jpg', 1)

    b,g,r = cv2.split(imgBGR)

    imgBGR_copy = imgBGR.copy()
##    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
##    cv2.imshow('Original image', imgBGR)
##    cv2.waitKey(0)

    'conver RGB into other channels'
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgLUV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LUV)
    imgHLS = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HLS)

    'select the one that you would like to try'
    select = int(raw_input('Enter choice (0 for BGR, 1 for HSV, 2 for LAB, 3 for LUV, 4 for HLS): '))
    if select == 0:
        img = imgBGR  # R is OK but not good
    elif select == 1:
        img = imgHSV  # H channel, 63-120 is perfect!!!!!
    elif select == 2:
        img = imgLAB  # A channel, 125-154 also perfect!!!
    elif select == 3:
        img = imgLUV  # U channel, 97-144 perfect!!!!
    elif select == 4:
        img = imgHLS  # H channel, 68-131 almost perfect

    'split it into 3 channels'
    c1,c2,c3 = cv2.split(img)

    ch1 = cv2.applyColorMap(c1, cv2.COLORMAP_JET)
    ch2 = cv2.applyColorMap(c2, cv2.COLORMAP_JET)
    ch3 = cv2.applyColorMap(c3, cv2.COLORMAP_JET)
    cv2.imshow('channel1', ch1)
    cv2.imshow('channel2', ch2)
    cv2.imshow('channel3', ch3)

    'create trackbar for tuning threshold'
    def nothing(*arg):
        pass
    cv2.namedWindow('control', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('minC1', 'control', 0, 256, nothing)
    cv2.createTrackbar('maxC1', 'control', 255, 256, nothing)
    cv2.createTrackbar('minC2', 'control', 0, 256, nothing)
    cv2.createTrackbar('maxC2', 'control', 255, 256, nothing)
    cv2.createTrackbar('minC3', 'control', 0, 256, nothing)
    cv2.createTrackbar('maxC3', 'control', 255, 256, nothing)
    cv2.namedWindow('selected_imgBGR', cv2.WINDOW_NORMAL)
    while(1):
        minC1 = cv2.getTrackbarPos('minC1', 'control')
        maxC1 = cv2.getTrackbarPos('maxC1', 'control')
        minC2 = cv2.getTrackbarPos('minC2', 'control')
        maxC2 = cv2.getTrackbarPos('maxC2', 'control')
        minC3 = cv2.getTrackbarPos('minC3', 'control')
        maxC3 = cv2.getTrackbarPos('maxC3', 'control')
        if (minC1 > maxC1):
            maxC1 = minC1
            cv2.setTrackbarPos('maxC1', 'control',maxC1)
        if (minC2 > maxC2):
            maxC2 = minC2
            cv2.setTrackbarPos('maxC2', 'control',maxC2)
        if (minC3 > maxC3):
            maxC3 = minC3
            cv2.setTrackbarPos('maxC3', 'control',maxC3)

        'generate the mask for each channel'
        c1_mask = cv2.inRange(c1, minC1, maxC1)
        c2_mask = cv2.inRange(c2, minC2, maxC2)
        c3_mask = cv2.inRange(c3, minC3, maxC3)
        c1_mask = np.logical_and(c1_mask,1)
        c2_mask = np.logical_and(c2_mask,1)
        c3_mask = np.logical_and(c3_mask,1)

        'combine the mask together'
        combined_mask = np.array(np.logical_and(np.logical_and(c1_mask, c2_mask), c3_mask)*255, np.uint8)
        'select subimage from each channel and then merge them together'
        selected_imgB = np.array(combined_mask/255.0 * b, np.uint8)
        selected_imgG = np.array(combined_mask/255.0 * g, np.uint8)
        selected_imgR = np.array(combined_mask/255.0 * r, np.uint8)
        selected_imgBGR = cv2.merge([selected_imgB,selected_imgG,selected_imgR])
        cv2.imshow('selected_imgBGR', selected_imgBGR)
        k = cv2.waitKey(1)
        if k == 27:
            print minC1, maxC1, minC2, maxC2, minC3, maxC3
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
