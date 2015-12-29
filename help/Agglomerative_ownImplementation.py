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

import cv2
import numpy as np
from TianUtility import *

"""
the cluster for the aggolomerate clustering algorithm
self.count is the number of elements in this cluster
self.components is the index for the elements in this cluster
self.clusterID is the ID for this cluster
"""
class Cluster:
    def __init__(self, label):
        self.count = 1
        self.components = set()
        self.components.add(label)
        self.clusterID = label
        self.weight = 0
        self.avgImg = None
        self.parentCluster = None

    def __repr__(self):
        return '(count %i, clusterID %i, components %s)' % (self.count, self.clusterID, self.components)

    def add(self, newmember):
        if self.parentCluster == None:
            self.components = self.components.union(newmember.components)
            newmember.clusterID = self.clusterID
            newmember.parentCluster = self
        else:
            self.parentCluster.components = self.parentCluster.components.union(newmember.components)
            newmember.clusterID = self.parentCluster.clusterID
            newmember.parentCluster = self.parentCluster

    def assignWeight(self, bigN):
        self.weight = float(self.count)/bigN

    def plot(self, colorPatches):
        print self
        # print "Cluster %i has %i components" % (self.clusterID, self.count)
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


"""
Implmentation of the agglomerative clustering. Each item is firstly by itself a cluster,
 and then nearby clusters are merged together.
 0.6 is finer clustering (better for visualization since they look more similar)
 0.4 is better for actual clustering to avoid small clusters
"""
def agglomerate(labels, grid, t = 0.6):
    L = len(labels)
    M, N = grid.shape
    assert (L == M == N)
    clusters = []
    # initial setup
    for i in range(L):
        clusters.append(Cluster(labels[i]))
    for i in range(N):
        for j in range(i+1, N):
            if grid[i][j] > t:
                clusters[i].add(clusters[j])
    clustersCount = 0
    relabelClusters = []
    for i in range(N):
        c = clusters[i]
        if c.parentCluster == None:
            c.clusterID = clustersCount
            c.count = len(c.components)
            clustersCount += 1
            relabelClusters.append(c)
    return relabelClusters

def main():
    grid = 1 - 2* np.random.rand(20,20)   # [-1, 1]
    names = range(20)
    agglomerate(names, grid)

if __name__ == '__main__':
    main()