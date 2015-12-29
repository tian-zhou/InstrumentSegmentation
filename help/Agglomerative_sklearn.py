# Author : Vincent Michel, 2010
#          Alexandre Gramfort, 2011
# License: BSD 3 clause

print(__doc__)

import time as time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import cv2
from TianUtility import *


def AggloCluster(X, _n_clusters = 5):
    clf = AgglomerativeClustering(n_clusters=_n_clusters, affinity='precomputed', linkage='average', ).fit(X)
    clusters = ConstructClustersFromLabel(clf.labels_)
    all_sum = GetSumDistance(clusters, X, _n_clusters)
    return clusters, all_sum

def GetSumDistance(clusters, X, numClusters):
    NumClusters = np.shape(X)[0]
    all_sum = 0
    for i in range(numClusters):
        c = clusters[i]
        minelement = np.min(c.components)
        c.sumDist = sum(X[minelement, c.components])
        all_sum += c.sumDist
    return all_sum


def ConstructClustersFromLabel(labels):
    NumClusters = np.max(labels) + 1
    clusters = []
    for i in range(NumClusters):
        memberIdx = np.nonzero(labels == i)[0]
        c = Cluster(i, memberIdx)
        clusters.append(c)
    return clusters

class Cluster:
    def __init__(self, id, memberIdx):
        self.components = memberIdx
        self.count = len(self.components)
        self.sumDist = 0
        self.clusterID = id
        self.weight = 0
        self.avgImg = None

    def __repr__(self):
        return '(clusterID %i, count %i, weight %f)' % (self.clusterID, self.count,self.weight)
        # return '(clusterID %i, count %i, weight %f, components %s)' % (self.clusterID, self.count,self.weight, self.components)


    def assignWeight(self, bigN):
        self.weight = float(self.count)/bigN

    def plot(self, colorPatches):
        print self
        enlargeFactor = 4
        bigN = 15 * enlargeFactor
        bigImage = np.zeros((bigN, bigN*self.count, 3), np.uint8)
        accumu = np.zeros(colorPatches[0].shape, np.float)
        for i in range(self.count):
            cv2.accumulate(colorPatches[self.components[i]], accumu)
            bigImage[:, i*bigN : (i+1)*bigN] = cv2.resize(colorPatches[self.components[i]], None, fx=enlargeFactor, fy=enlargeFactor,interpolation = cv2.INTER_CUBIC)

        self.avgImg = (accumu/self.count).astype(np.uint8)
        cv2.imshow('Cluster'+str(self.clusterID), bigImage)
        cv2.imshow('Rescaled Average image'+str(self.clusterID), cv2.resize(self.avgImg, None, fx=enlargeFactor, fy=enlargeFactor,interpolation = cv2.INTER_CUBIC))
        WaitKey(0)


def main():
    ###############################################################################
    # Generate data
    np.set_printoptions(precision=2)
    N = 10
    X = 1 - 2 * np.random.rand(N,N)   # [-1, 1]
    X = 1-X
    i,j = np.triu_indices(X.shape[0], k=1)
    Y = np.zeros((N,N))
    Y[i,j] = X[i,j]
    print Y
    # only the upper right triangle will work

    ###############################################################################
    # Compute clustering
    clusters = AggloCluster(Y)
    print 1

if __name__ == '__main__':
    main()