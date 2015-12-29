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
#from sklearn_Agglomerative import *
from TianUtility import *
from time import time
from sklearn import mixture
import sys
sys.path.append('.\morphsnakes-master')
import morphsnakes

class Codebook:
    def __init__(self):
        pass

    """
    Read in the color, depth and mask image from the training data set
    return the readin images
    """
    def ReadImage(self, fileIndex = 1, show = False):
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

    """
    rescale the depth and mask image to have the same size as color image
    here the height ratio is actually different with width ratio, with a small
    difference which we will ignore.
    """
    def RescaleDepth(self, imgBGR, imgDepth, imgMask):
        h,w,_ = imgBGR.shape
        imgDepth = cv2.resize(imgDepth,(w, h), interpolation = cv2.INTER_CUBIC)
        imgMask = cv2.resize(imgMask,(w, h), interpolation = cv2.INTER_CUBIC)
        return imgDepth, imgMask

    """
    apply the mask to the input image
    """
    def ApplyMask(self, img, imgMask):
        return cv2.bitwise_and(img, img, mask = imgMask)

    """
    Normalize the image so that it is illumination invariant
    Method 1: histogram equalization
    Method 2: normalize the grayscale value directly (does not work)
    """
    def IlluminationNorm(self, imgGray, show = False):
        # method 1
        imgGrayHistEqu = cv2.equalizeHist(imgGray)

        # method 2
        # imgGrayNorm = imgGray.copy()
        # cv2.normalize(imgGrayNorm, imgGrayNorm, 0, 255, cv2.NORM_MINMAX) # the same
        # cv2.normalize(imgGrayNorm.astype(np.float), imgGrayNorm, 1)
        # cv2.normalize(imgGrayNorm, imgGrayNorm, 0, 255, cv2.NORM_MINMAX) # the same

        # display
        if show == True:
            cv2.imshow("imgGray", imgGray)
            cv2.imshow("imgGrayHistEqu", imgGrayHistEqu)
            #cv2.imshow("imgGrayNorm", imgGrayNorm)
            WaitKey(1)
        return imgGrayHistEqu

    """
    Apply non-maximum suppresion to avoid duplicate positive responses in a near window.
    If the positive response is not the largest in a neighboring window (WxW), it will
    be assigned 0. We only keep the largest.

    THIS FUNCTION RUNS SLOW!
    THREE LOOPS!
    """
    def NonMaximumSuppresion(self, imgMask, imgResponse, large = True, winsize = 15):
        startTime = time()
        imgMaskCp = imgMask.copy()
        M,N = imgMaskCp.shape
        halfwinsize = int(winsize/2)
        for i in range(halfwinsize, M-halfwinsize):
            for j in range(halfwinsize, N-halfwinsize):
                if imgMaskCp[i,j] == 255:
                    if large == True:
                        if imgResponse[i,j] != \
                        np.amax(imgResponse[i-halfwinsize:i+halfwinsize+1,\
                        j-halfwinsize:j+halfwinsize+1]):
                            imgMaskCp[i,j] = 0
                    elif large == False:
                        if imgResponse[i,j] != \
                        np.amin(imgResponse[i-halfwinsize:i+halfwinsize+1,\
                        j-halfwinsize:j+halfwinsize+1]):
                            imgMaskCp[i,j] = 0
        elapsedTime = time() - startTime
        print "NonMaximumSuppresion function takes %f seconds to finish" % elapsedTime
        return imgMaskCp

    """
    Get the Harris Corner responce matrix, also threshold it with min or max
    @ large: if True gets the large responce (corners)
            if False gets the small response (plain area)
    @ tau: the threshold = tau * response.max(), 1e-6 for small and 1e-2 for large

    if edge,    response -> -inf
    if cornder, response -> +inf
    if plain,   response -> 0
    we want both edges and corners, so we threshold on abs(response)
    """
    def HarrisCorner(self, imgBGR, imgGray, imgMask, large = True, tau = 1e-2, show = True, NMS = True):
        imgGray = np.float32(imgGray)
        response = cv2.cornerHarris(imgGray,blockSize = 2, ksize = 3, k = 0.04)
        responseAbs = abs(response)

        # threshold the response
        if large == True:
            _,cornerMask = cv2.threshold(responseAbs,tau*responseAbs.max(),255,cv2.THRESH_BINARY)
            # apply mask to select only foreground
            cornerMask = self.ApplyMask(cornerMask, imgMask)
        elif large == False:
            _,cornerMask = cv2.threshold(responseAbs,tau*responseAbs.max(),255,cv2.THRESH_BINARY_INV)
            cornerMask = self.ApplyMask(cornerMask, imgMask)

        # median blue
        cornerMask = cornerMask.astype(np.uint8)
        if large == True:
            cornerMask = cv2.medianBlur(cornerMask, 3)
        else:
            cornerMask = cv2.medianBlur(cornerMask, 7)

        # non-maximum suppresion
        if NMS == True:
            cornerMaskNMS = self.NonMaximumSuppresion(cornerMask, responseAbs, large)
        else:
            cornerMaskNMS = cornerMask.copy()

        cornerMaskNMSDilate = cv2.dilate(cornerMaskNMS, None)

        # rescale responseAbs for better visualization, it is a float image
        cv2.normalize(responseAbs, responseAbs, 0, 1, cv2.NORM_MINMAX)
        if show == True:
            imgBGRShow = imgBGR.copy()
            imgBGRShow[cornerMaskNMSDilate == 255]=[0,0,255]
            cv2.imshow('responseAbs', responseAbs)
            cv2.imshow('imgBGR with corner', imgBGRShow)
            cv2.imshow('cornerMask', cornerMask)
            cv2.imshow('cornerMaskNMSDilate', cornerMaskNMSDilate)
            WaitKey(0)
        return cornerMaskNMS

    """
    display the two masks for foreground and background together on the image
    """
    def ShowTwoMasksTogether(self, imgBGR, fgMask, bgMask):
        imgBGRShow = imgBGR.copy()
        imgBGRShow[fgMask == 255]=[255,255,0]
        imgBGRShow[bgMask == 255]=[0,0,255]
        cv2.imshow('imgBGR with corner', imgBGRShow)
        WaitKey(0)


    """
    Extract patches based on masks. For each non-zero element in the mask,
    we extract the WxW patch surrounding it
    """
    def ExtractPatches(self, imgBGR, cornerMask, W = 15):
        M,N,_ = imgBGR.shape
        halfW = int(W/2)
        colorPatches = []
        for i in range(halfW, M-halfW):
            for j in range(halfW, N-halfW):
                if cornerMask[i,j] == 255:
                    colorPatch = imgBGR[i-halfW: i+halfW+1, j-halfW: j+halfW+1]
                    colorPatches.append(colorPatch)
        return colorPatches

    """
    resize the color patches and visualize them
    """
    def VisPatches(self, colorPatches):
        N = len(colorPatches)
        print "Display %d patches." % N
        Nsqrt = int(np.sqrt(N))
        enlargeFactor = 4
        bigN = 15 * enlargeFactor
        bigImage = np.zeros((bigN*Nsqrt, bigN*Nsqrt, 3), np.uint8)
        for i in range(Nsqrt**2):
            bigImage[(i % Nsqrt) * bigN: (i % Nsqrt) * bigN + bigN, (i/Nsqrt) * bigN: (i/Nsqrt) * bigN+bigN, :] \
            = cv2.resize(colorPatches[i], None, fx=enlargeFactor, fy=enlargeFactor,interpolation = cv2.INTER_CUBIC)
            #cv2.imshow('Patch'+str(i+1), cv2.resize(colorPatches[i], None, fx=2, fy=2,interpolation = cv2.INTER_CUBIC))
            #cv2.moveWindow('Patch'+str(i+1), (i % Nsqrt) * 30, (i/Nsqrt) * 30)
            #WaitKey(10)
        cv2.imshow('Extracted patches', bigImage)
        WaitKey(0)

    """
    calculate the NGC between two image patches
    it should be a value in [-1,1] to evaluate the similarity between
    two images
    NGC =
    +1 : same image
    -1 : entirely different image
     0 : no relation
    """
    def NGC(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float)
        M, N = img1.shape
        M2, N2 = img2.shape
        assert (M == M2 and N == N2)
        numerator = np.sum(np.multiply(img1 - np.mean(img1), img2-np.mean(img2)))
        denominator = M * np.std(img1) * M * np.std(img2)
        ngc = numerator/denominator
        ngc = min(ngc, 1)
        assert (abs(ngc) <= 1)
        return ngc

    """
    Given a color patches list, generate the similarity matrix within the list
    The similarity measure is NGC between two images
    RUNS SLOW!!!
    NGC takes time and this is O(N^2)
    """
    def GetSimilarityMatrix(self, colorPatches):
        start_time = time()
        N = len(colorPatches)
        simi = np.zeros((N,N), np.float)
        for i in range(N):
            for j in range(i+1, N):
                simi[i,j] = self.NGC(colorPatches[i], colorPatches[j])
        elapsed_time = time() - start_time
        print "GetSimilarityMatrix function takes %f seconds to finish" % elapsed_time
        return simi

    """
    codebook generation from one image
    """
    def CodebookGenerationOneImage(self, imgBGR, imgGrayHistEqu, imgMask, _large, N_clusters = 10):
        if _large == True:
            Mask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = True, tau = 1e-2)
        else:
            Mask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = False, tau = 1e-6)

        ColorPatches = self.ExtractPatches(imgBGR, Mask)
        self.VisPatches(ColorPatches)

        Simi = self.GetSimilarityMatrix(ColorPatches)
        Dist = 1-Simi

        Clusters, SumDist = AggloCluster(Dist, _n_clusters = N_clusters)

        for i in range(len(Clusters)):
            Clusters[i].assignWeight(len(ColorPatches))
            Clusters[i].plot(ColorPatches)

        return ColorPatches, len(ColorPatches), Clusters

    """
    codebook generation from the input color features
    the colorpatches are used to plot the elements in the
    clustering better
    """
    def CodebookGeneration(self, ColorFeatures, ColorPatches, _N_comp = 10, show = False):
        obs = ColorFeatures
        GMM = mixture.GMM(n_components=_N_comp)
        GMM.fit(obs)
        Labels = GMM.predict(obs)
        Clusters = []
        for i in range(_N_comp):
            c = Cluster(i, np.nonzero(Labels == i)[0], ColorPatches)
            c.assignWeight(GMM.weights_[i])
            if show == True:
                c.plot(ColorPatches)
            Clusters.append(c)
        return Clusters, GMM

    """
    evaluate the training fg and bg GMMs
    """
    def EvaluateGMM(self, fgGMM, bgGMM, fgColorFeatures, bgColorFeatures):
        # Evaluate GMM result
        o = fgColorFeatures
        fgLikelihood = fgGMM.score(o)
        bgLikelihood = bgGMM.score(o)
        pred = fgLikelihood > bgLikelihood
        print "fg accuracy: ", np.round(np.sum(pred)/float(len(fgLikelihood)), 3)

        o = bgColorFeatures
        fgLikelihood = fgGMM.score(o)
        bgLikelihood = bgGMM.score(o)
        pred = fgLikelihood < bgLikelihood
        print "bg accuracy: ", np.round(np.sum(pred)/float(len(bgLikelihood)), 3)


    """
    Shift the NGCs from range [-1,1] to [0,2] so that they are all positive.
    Then normalize all the NGCs into a probability distribution
    """
    def NormalizeNGCs(self, NGCs):
        NGCs += 1
        NGCs /= np.sum(NGCs)
        return NGCs

    """
    for each observation, estimate P(o|H)
    P(o|H) = sum_j {P(o|I_j, H) * P(I_j| H)}
    """
    def CalcLikelihood(self, o, ColorClusters):
        NGCs = np.zeros(len(ColorClusters))
        for j in range(len(ColorClusters)):
            NGCs[j] = self.NGC(o, ColorClusters[j].avgImg)

        """
        1, uniform patch, bad
        """
##        sumP = np.sum(NGCs + 1)

        """
        3, full average, worst
        """
##        sumP = np.sum([NGCs[i] * ColorClusters[i].weight for i in range(len(ColorClusters))])


        """
        4, max pooling, good
        """
        # maxP = np.amax([NGCs[i] * ColorClusters[i].weight for i in range(len(ColorClusters))])

        """
        5, max pooling, best
        """
        maxP = np.amax(NGCs)
        return maxP


    """
    Read the saved color patches, for background or foreground
    """
    def ReadColorPatches(self, fg = True, _N = 100):
        ColorPatches = []
        if fg == True:
            N = min(_N, 535) # in total we have 535 fg images
            for i in range(N):
                ColorPatches.append(cv2.imread(".\\Patches\\foreground\\fg_"+str(i) + ".jpg", 1))
        elif fg == False:
            N = min(_N, 1131) # in total we have 1131 bg images
            for i in range(N):
                ColorPatches.append(cv2.imread(".\\Patches\\background\\bg_"+str(i) + ".jpg", 1))
        return ColorPatches


    """
    create features for the color patches
    """
    def ColorFeature(self, ColorPatches, method = 'gray'):
        N = len(ColorPatches)
        W, _, _ = ColorPatches[0].shape
        if method == 'gray':
            # stack the gray values
            features = np.zeros((N, W**2))
            for i in range(N):
                features[i,:] = np.ravel(cv2.cvtColor(ColorPatches[i], cv2.COLOR_BGR2GRAY))
        if method == 'hist':
            # histogram for the grayscale images
            Nbins = 16
            features = np.zeros((N, Nbins))
            for i in range(N):
                features[i,:] = np.ravel(cv2.calcHist([cv2.cvtColor(ColorPatches[i], cv2.COLOR_BGR2GRAY)], [0], None, [Nbins], [0, 256]))
        if method == 'hog':
            # hog for the 15x15 patch
            pass
        if method == 'sift':
            pass
        if method == 'surf':
            pass

        return features

    """
    Detect and draw SIFT features on a given image
    """
    def SIFT(self, imgBGR, show = True):
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(imgGray,None)
        # the SIFT descriptor is 128 dimension
        imgBGRcp = imgBGR.copy()
        imgBGRcp = cv2.drawKeypoints(imgBGRcp,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('img_SIFT', imgBGRcp)
        WaitKey(0)

    """
    Detect and draw SURF features on a given image
    It seems that SURF is easier to play with since it is easier to set parameters
    """
    def SURF(self, imgBGR, show = True):
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(4000) # small number -> more salient points
        kp, des = surf.detectAndCompute(imgGray,None)
        # the SIFT descriptor is 128 dimension
        imgBGRcp = imgBGR.copy()
        imgBGRcp = cv2.drawKeypoints(imgBGRcp,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('img_SURF', imgBGRcp)
        WaitKey(0)

    """
    check the SIFT/SURF interesting points for the input image
    """
    def SIFT_SURF_test(self, imgBGR):
        self.SIFT(imgBGR)
        self.SURF(imgBGR)

    """
    experiment 1:
    test the opencv sift and surf interesting point extractor
    visualize them to see how good they can capture the foreground and background
    """
    def exp1(self, imgBGR):
        self.SIFT_SURF_test(imgBGR)

    """
    experiment 2:
    show masks for both foreground and background.
    need to disable the non-maximum suppression in HarrisCorner function
    for better visualization
    """
    def exp2(self, imgBGR, imgGrayHistEqu, imgMask):
        fgMask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = True, tau = 1e-2, NMS = False)
        bgMask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = False, tau = 1e-6, NMS = False)
        self.ShowTwoMasksTogether(imgBGR, fgMask, bgMask)

    """
    experiment 3:
    test differnt cluster size for the aggomerative clustering
    """
    def exp3(self, imgBGR, imgGrayHistEqu, imgMask, _large):
        if _large == True:
            Mask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = True, tau = 1e-2)
        else:
            Mask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = False, tau = 1e-6)

        ColorPatches = self.ExtractPatches(imgBGR, Mask)
        self.VisPatches(ColorPatches)
        Simi = self.GetSimilarityMatrix(ColorPatches)
        Dist = 1-Simi

        numClusRange = range(5, 100, 5)
        SumDist = np.zeros((len(numClusRange)))
        times = np.zeros((len(numClusRange)))
        for i in range(len(numClusRange)):
            startTime = time()
            Clusters, SumDist[i] = AggloCluster(Dist, numClusRange[i])
            elapsedTime = time() - startTime
            times[i] = elapsedTime
            # print "number of cluters: %i, sum Distance: %f" % (numClusRange[i], SumDist[i])

        plt.figure(1)
        plt.plot(numClusRange, SumDist)
        plt.title("experiemnt with number of clusters in the agglomerative clustering")
        plt.xlabel("number of clusters")
        plt.ylabel("sum of distances within clusters (large -> bad cluster)")

        plt.figure(2)
        plt.plot(numClusRange, times)
        plt.title("experiemnt with number of clusters in the agglomerative clustering")
        plt.xlabel("number of clusters")
        plt.ylabel("elapsed time for convergence (in seconds)")
        plt.show()

    """
    experiment 4:
    agglomerative clustering + NGC
    """
    def exp4(self):
        for fileIndex in range(1,21):
            imgBGR, imgGray, imgGrayHistEqu, imgDepth, imgMask = self.ReadImage(fileIndex)

            # codebook generation
            fgColorPatches, fgColorPatchesN, fgColorClusters = self.CodebookGenerationOneImage(imgBGR, imgGrayHistEqu, imgMask, _large = True, N_clusters = 40) # 40 is better than 10
            bgColorPatches, bgColorPatchesN, bgColorClusters = self.CodebookGenerationOneImage(imgBGR, imgGrayHistEqu, imgMask, _large = False, N_clusters = 40)

            # for each observation, estimate P(o|I_j, H_1)
            pred = np.zeros(fgColorPatchesN)
            for i in range(fgColorPatchesN):
                o = fgColorPatches[i]
                fgLikelihood = self.CalcLikelihood(o, fgColorClusters)
                bgLikelihood = self.CalcLikelihood(o, bgColorClusters)
                pred[i] = fgLikelihood > bgLikelihood
                #print "fgLikelihood: %.2f, bgLikelihood %.2f" % (fgLikelihood, bgLikelihood)
            print "accuracy: ",np.sum(pred)/float(fgColorPatchesN)

            pred = np.zeros(bgColorPatchesN)
            for i in range(bgColorPatchesN):
                o = bgColorPatches[i]
                fgLikelihood = self.CalcLikelihood(o, fgColorClusters)
                bgLikelihood = self.CalcLikelihood(o, bgColorClusters)
                pred[i] = fgLikelihood < bgLikelihood
                #print "fgLikelihood: %.2f, bgLikelihood %.2f" % (fgLikelihood, bgLikelihood)
            print "accuracy: ",np.sum(pred)/float(bgColorPatchesN)
            # end the program


    """
    save the local color patches for both fg and bg
    """
    def SaveLocalColorPatches(self):
        for fileIndex in range(1,21):
            imgBGR, imgGray, imgGrayHistEqu, imgDepth, imgMask = self.ReadImage(fileIndex)

            global GLOBAL_numbg, GLOBAL_numfg

            fgMask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = True, tau = 1e-2)
            fgColorPatches = self.ExtractPatches(imgBGR, fgMask)
            for i in range(len(fgColorPatches)):
                cv2.imwrite(".\\Patches\\foreground\\fg_"+str(i+GLOBAL_numfg) + ".jpg", fgColorPatches[i])
            GLOBAL_numfg = len(fgColorPatches)

            bgMask = self.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = False, tau = 1e-6)
            bgColorPatches = self.ExtractPatches(imgBGR, bgMask)
            for i in range(len(bgColorPatches)):
                cv2.imwrite(".\\Patches\\background\\bg_"+str(i+GLOBAL_numbg) + ".jpg", bgColorPatches[i])
            GLOBAL_numbg = len(bgColorPatches)

    """
    Given an input image and a histogram, do the histogram backprojection
    return a mask
    """
    def HistBackProj(self, imgBGR, target_hist, show = True):
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        cv2.normalize(target_hist,target_hist,0,255,cv2.NORM_MINMAX)
        prob = cv2.calcBackProject([imgHSV],[0,1], target_hist,[0,180,0,256],1)
        if show:
            cv2.imshow('prob', prob)

        # Now convolute with circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cv2.filter2D(prob,-1,disc,prob)
        if show:
            cv2.imshow('prob2', prob)

        # threshold and binary AND
        ret, mask = cv2.threshold(prob,128,255,0)
        mask = cv2.merge((mask,mask,mask))
        ROIimgBGR = cv2.bitwise_and(imgBGR,mask)
        if show:
            cv2.imshow('mask', mask)
            cv2.imshow('ROIimgBGR', ROIimgBGR)
        WaitKey(1)
        return prob, mask, ROIimgBGR

    """
    use contour manipulation to get rid of the small components in the foreground
    """
    def ContourClean(self, mask):
        _, BW = cv2.threshold(mask,127,255,0)
        contours, hierarchy = cv2.findContours(BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        cntImg = np.zeros(mask.shape, np.uint8)
        for i in range(len(contours)):
            cnt = contours[i]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            if i == 0:
                areaT = 0.05 * area
            if area > areaT:
                #print "\nContour index %i" % i
                #print "This area: %f" % area

                cv2.drawContours(cntImg, contours, i, 255, -1)
                cv2.imshow('cntImg', cntImg)
            WaitKey(1)
        return cntImg

    """
    smoothing:
        large, the boundary is smoother
        small, the boundary is coarse
        good: 5
    threshold:
        small, keep shrinking and will vanish
        large, does not move at all
        good: 0.3
    balloon:
        <0, contour become smaller
        >0, contour becomes larger
        good: -1
    """
    def snake(self, imgBGR, imgMask):
        # Load the image.
        #imgcolor = imread("../training/color_1.jpg")/255.0
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        imgBGR = (imgBGR/255.0).astype(np.float)
        imgGray =(imgGray/255.0).astype(np.float)

        # g(I)
        gI = morphsnakes.gborders(imgGray, alpha=1000, sigma=2)

        # Morphological GAC. Initialization of the level-set.
        mgac = morphsnakes.MorphGAC(gI, smoothing=10, threshold=0.3, balloon=-1)
        _, imgMask = cv2.threshold(imgMask,127,255,cv2.THRESH_BINARY)
        mgac.levelset = imgMask

        # plot
        cv2.imshow('imgBGR', imgBGR)
        cv2.imshow('imgGray', imgGray)
        cv2.imshow('gI', gI)
        cv2.imshow('mgac.levelset', mgac.levelset)
        cv2.waitKey(0)

        # Visual evolution.
        plt.figure()
        b,g,r = cv2.split(imgBGR)
        imgRGB = cv2.merge([r,g,b])
        morphsnakes.evolve_visual(mgac, num_iters=10, background=imgRGB)
        return mgac.levelset


GLOBAL_numfg = 0
GLOBAL_numbg = 0


class Cluster:
    def __init__(self, id, components, colorPatches):
        self.components = components
        self.count = len(self.components)
        self.clusterID = id
        self.weight = 0
        self.avgImg = None
        self.bigImage= None
        self.hist = None
        self.enlargeFactor = 6
        self.summarize(colorPatches)


    def summarize(self, colorPatches):
        # get average image
        accumu = np.zeros(colorPatches[0].shape, np.float)
        for i in range(self.count):
            cv2.accumulate(colorPatches[self.components[i]], accumu)
        self.avgImg = (accumu/self.count).astype(np.uint8)

        # get hist
        W,H,_ = colorPatches[0].shape
        self.bigImage = np.zeros((W, H*self.count, 3), np.uint8)
        for i in range(self.count):
           self. bigImage[:, i*H : (i+1)*H, :] = colorPatches[self.components[i]]
        bigImageHSV = cv2.cvtColor(self.bigImage, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([bigImageHSV], [0,1], None, [180, 256],[0, 180,0,256])

        # self.plot(colorPatches)
        # get pca image
        # it seems that the PCA is not as good as the average image to visualize the cluster
        # self.CalcPCA(colorPatches)

    def CalcPCA(self, colorPatches, K = 1, show = True):
        width, height, _ = colorPatches[0].shape
        N = len(self.components)

        X = np.zeros((width*height*3 , N))
        for i in range(N):
            #x_vec = np.ravel(colorPatches[i][:,:,0]).astype(float)
            x_vec = np.ravel(colorPatches[self.components[i]]).astype(float)
            #x_vec = np.ravel(cv2.cvtColor(colorPatches[self.components[i]], cv2.COLOR_BGR2GRAY)).astype(float)
            x_vec /= np.linalg.norm(x_vec)
            X[:,i] = x_vec
        GrandMean = np.mean(X,1)
        for i in range(N):
            X[:,i] -= GrandMean
        U,s,Vt = np.linalg.svd(np.dot(np.transpose(X), X))
        Wp = np.zeros((width*height*3, K))
        for i in range(K):
            wi = np.dot(X,U[:,i])
            wi /= np.linalg.norm(wi)
            Wp[:,i] = wi

        # Visualize the most significant Eigen-Patterns
        if show == True:
            for k in range(K):
                w = np.reshape(Wp[:,k]+GrandMean, (width,height,3)).astype(np.float)
                cv2.normalize(w, w, 0, 255, cv2.NORM_MINMAX) # the same
                w = w.astype(np.uint8)
                #wint_color = cv2.applyColorMap(wint, cv2.COLORMAP_JET)
                cv2.imshow('w'+str(k),cv2.resize(w, None, fx=self.enlargeFactor, fy=self.enlargeFactor, interpolation = cv2.INTER_CUBIC))
                #cv2.imshow('wint_color'+str(k),cv2.resize(wint_color, None, fx=self.enlargeFactor, fy=self.enlargeFactor, interpolation = cv2.INTER_CUBIC))
                WaitKey(0)

    def __repr__(self):
        return '(clusterID %i, count %i, weight %f)' % (self.clusterID, self.count,self.weight)

    def assignWeight(self, weight):
        self.weight = weight

    def plot(self, colorPatches):
        print self
        cv2.imshow('Cluster'+str(self.clusterID), cv2.resize(self.bigImage, None, fx=self.enlargeFactor, fy=self.enlargeFactor,interpolation = cv2.INTER_CUBIC))
        cv2.imshow('Rescaled Average image'+str(self.clusterID), cv2.resize(self.avgImg, None, fx=self.enlargeFactor, fy=self.enlargeFactor,interpolation = cv2.INTER_CUBIC))
        WaitKey(1)


def main():
    cb = Codebook()

    """
    GMM to fit the color features for foreground and background
    """
    N_comp = 10 # surprisingly, 10 is the best...
    fgColorPatches = cb.ReadColorPatches(fg = True, _N = 2000)
    fgColorFeatures = cb.ColorFeature(fgColorPatches, method = 'gray')
    fgClusters, fgGMM = cb.CodebookGeneration(fgColorFeatures, fgColorPatches, _N_comp = N_comp)

    bgColorPatches = cb.ReadColorPatches(fg = False, _N = 2000)
    bgColorFeatures = cb.ColorFeature(bgColorPatches, method = 'gray')
    bgClusters, bgGMM = cb.CodebookGeneration(bgColorFeatures, bgColorPatches, _N_comp = N_comp)

    """
    Evaluate GMM result on the same training data
    TO-DO, should split the data into training and validation
    """
    cb.EvaluateGMM(fgGMM, bgGMM, fgColorFeatures, bgColorFeatures)

    """
    segmentation for each image using the training GMMs
    """
    for fileIndex in range(1,21):
        imgBGR, imgGray, imgGrayHistEqu, imgDepth, imgMask = cb.ReadImage(fileIndex)

        """
        !!!
        """
        enableWeight = True

        """
        histogram back projection for foreground
        """
        fgProbs = np.zeros((imgBGR.shape[0:2]), np.float)
        for i in range(N_comp):
            prob, mask, ROIimgBGR = cb.HistBackProj(imgBGR, fgClusters[i].hist, show=False)
            if enableWeight == False:
                fgProbs += prob
            else:
                fgProbs += prob.astype(np.float) * fgClusters[i].weight
        fgProbs = np.divide(fgProbs, N_comp).astype(np.uint8)
        #cv2.imshow('fgProbs', fgProbs)
        #ret, fgMask = cv2.threshold(fgProbs,180,255,0)
        ret, fgMask = cv2.threshold(fgProbs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        fgMask = cb.ApplyMask(fgMask, imgMask)
        fgMask = cv2.medianBlur(fgMask, 5)
        fgMask3Channel = cv2.merge((fgMask,fgMask,fgMask))
        fgimgBGR = cv2.bitwise_and(imgBGR,fgMask3Channel)
        #cv2.imshow('fgMask', fgMask)
        cv2.imshow('fgimgBGR', fgimgBGR)

        """
        histogram back projection for background
        """
        bgProbs = np.zeros((imgBGR.shape[0:2]), np.float)
        for i in range(N_comp):
            prob, mask, ROIimgBGR = cb.HistBackProj(imgBGR, bgClusters[i].hist, show=False)
            if enableWeight == False:
                bgProbs += prob
            else:
                bgProbs += prob.astype(np.float) * bgClusters[i].weight
        bgProbs = np.divide(bgProbs, N_comp).astype(np.uint8)
        #cv2.imshow('bgProbs', bgProbs)
        #ret, bgMask = cv2.threshold(bgProbs,100,255,0)
        ret, bgMask = cv2.threshold(bgProbs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bgMask = cb.ApplyMask(bgMask, imgMask)
        bgMask = cv2.medianBlur(bgMask, 5)
        bgMask3Channel = cv2.merge((bgMask,bgMask,bgMask))
        bgimgBGR = cv2.bitwise_and(imgBGR,bgMask3Channel)
        #cv2.imshow('bgMask', bgMask)
        cv2.imshow('bgimgBGR', bgimgBGR)

        """
        ratio of fg prob over bg prob
        """
        bgProbs = (bgProbs + 0.1).astype(np.float)
        ratio = np.divide(fgProbs.astype(np.float), bgProbs)
        rho = 3
        combMask = (255 * (ratio > rho)).astype(np.uint8)
        combMask = cb.ApplyMask(combMask, imgMask)
        combMask = cv2.medianBlur(combMask, 9)
        cv2.imshow('combMask', combMask)
        combMask3Channel = cv2.merge((combMask,combMask,combMask))
        combimgBGR = cv2.bitwise_and(imgBGR,combMask3Channel)
        cv2.imshow('combimgBGR', combimgBGR)

        """
        dilate the mask and then use snake algorithm
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        combMaskBig = cv2.dilate(combMask, kernel, iterations = 3)
        cv2.imshow('combMaskBig', combMaskBig)
        combMaskBig3Channel = cv2.merge((combMaskBig,combMaskBig,combMaskBig))
        combimgBGRBig = cv2.bitwise_and(imgBGR, combMaskBig3Channel)
        cv2.imshow("combimgBGRBig", combimgBGRBig)

        """
        use contour manipulation to clean the fg mask
        """
        cleanMask = cb.ContourClean(combMaskBig)
        cv2.imwrite('./intermediate/mask_'+str(fileIndex)+'.jpg', cleanMask)
        WaitKey(0)

        """
        snake!!!
        """
        snakeMask = cb.snake(imgBGR, cleanMask)
        continue


        """
        use the same Harris corner detector to find the salient points
        for both fg and bg, then use the GMM model to find out if the
        patches belong to foreground or background
        This is going to work very well. I think.
        But this is over-fitting to our data. It is not right.
        """
        fgMask = cb.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = True, tau = 1e-2)
        fgColorPatches = cb.ExtractPatches(imgBGR, fgMask)
        fgColorFeatures = cb.ColorFeature(fgColorPatches, method = 'gray')
        o = fgColorFeatures
        fgLikelihood = fgGMM.score(o)
        bgLikelihood = bgGMM.score(o)
        pred = fgLikelihood > bgLikelihood
        fgColorModel = [fgColorPatches[i] for i in range(len(fgColorPatches)) if pred[i] == 1]


        bgMask = cb.HarrisCorner(imgBGR, imgGrayHistEqu, imgMask, large = False, tau = 1e-6)
        bgColorPatches = cb.ExtractPatches(imgBGR, bgMask)
        bgColorFeatures = cb.ColorFeature(bgColorPatches, method = 'gray')
        o = bgColorFeatures
        fgLikelihood = fgGMM.score(o)
        bgLikelihood = bgGMM.score(o)
        pred = fgLikelihood < bgLikelihood
        bgColorModel = [bgColorPatches[i] for i in range(len(bgColorPatches)) if pred[i] == 1]

        print 1;


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
