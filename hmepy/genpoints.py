import numpy as np
import matplotlib.pyplot as plt
import random
import math

def runifs(n, d, c = None, r = 1):
    if c is None:
        c = np.zeros(d)
    initialPoints = np.random.normal(0, 1, (n,d))
    expVals = np.random.exponential(1, n)
    denoms = np.sqrt(expVals + np.apply_along_axis(lambda x: np.sum(x**2), 1, initialPoints))
    swept = np.apply_along_axis(lambda x: x/denoms, 0, initialPoints)
    return np.apply_along_axis(lambda x: x+c, 1, swept*r)

def punifs(x, c = None, r = 1):
    if c is None:
        c = np.zeros(len(x))
    testVal = np.sum((x-c)**2)/r**2
    if testVal <= 1: return 1
    return 0

def inRange(data, ranges):
    lowerBounds = [ranges[rkey][0] for rkey in list(ranges.keys())]
    upperBounds = [ranges[rkey][1] for rkey in list(ranges.keys())]
    isBigger = np.apply_along_axis(lambda x: all([x[i] >= lowerBounds[i] for i in range(len(x))]), 1, data)
    isSmaller = np.apply_along_axis(lambda x: all([x[i] <= upperBounds[i] for i in range(len(x))]), 1, data)
    return isBigger & isSmaller

def maximinSample(points, n, reps = 1000):
    sampleList = np.zeros((reps, n))
    measureList = np.zeros(reps)
    for i in range(reps):
        sampleIndex = random.sample(range(np.shape(points)[0]), n)
        sampledVals = points[sampleIndex,:]
        measure = [math.dist(sampledVals[i], sampledVals[j]) for i in range(sampledVals.shape[0]) for j in np.arange(start = i+1, stop = sampledVals.shape[1])]
        sampleList[i,:] = sampleIndex
        measureList[i] = min(measure)
    bestIndices = sampleList[np.argmax(measureList),:]
    return points[[int(a) for a in bestIndices],:]

# testrunif = runifs(1000, 2)
# plt.scatter(testrunif[:,0], testrunif[:,1], s = 0.5)
#plt.show()
# testMaxi = maximinSample(testrunif, 200)
# plt.scatter(testMaxi[:,0], testMaxi[:,1], s = 0.7)
# plt.show()
