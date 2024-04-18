import numpy as np
import pandas as pd
import math
import warnings
from copy import deepcopy
# Production
from hmepy.emulator import *
# Testing
#from emulator import *

def sequentialImp(ems, x, z, n = 1, cutoff = 3):
    outres = np.full(x.shape[0], True)
    for i in range(x.shape[0]):
        howManyFails = 0
        for j in list(ems.keys()):
            thisEm = ems[j]
            thisTarg = z[j]
            thisPoint = pd.DataFrame(x, index = [i], columns = x.columns)
            thisImp = sum(thisEm.implausibility(thisPoint, thisTarg, cutoff))
            if thisImp == False:
                howManyFails = howManyFails + 1
                if howManyFails >= n:
                    outres[i] = False
                    break
        if outres[i] == False:
            continue
    return outres

def nthImplausible(ems, x, z, n = None, maxImp = math.inf,
                   cutoff = None, sequential = False, getRaw = False):
    if isinstance(ems, Emulator):
        if not ems.outputName in list(z.keys()):
            raise KeyError("Target not found corresponding to named emulator.")
        return ems.implausibility(x, z[ems.outputName], cutoff)
    emNames = np.unique([ems[i].outputName for i in list(ems.keys())])
    if n is None:
        if len(emNames) > 10:
            n = 2
        else:
            n = 1
    for key in list(z.keys()):
        if len(z[key]) == 1:
            warnings.warn("Target " + key + " is a single value; assuming it's a val with 5\% sigma.")
            z[key] = {'val': z[key], 'sigma': z[key]*0.05}
    if n > len(emNames):
        warnings.warn("n cannot be greater than the number of targets. Switching to minimum implausibility.")
        n = len(emNames)
    if not(cutoff is None) and len(cutoff == 1):
        cutoff = np.full(len(ems), cutoff)
    if not(cutoff is None) and (len(ems) > 10 or sequential):
        return sequentialImp(ems, x, z, n, cutoff[0])
    implausibles = np.c_[[ems[key].implausibility(x, z[key], cutoff) for key in list(ems.keys())]].T
    dImplausibles = pd.DataFrame(implausibles, index = range(implausibles.shape[0]), columns = [ems[key].outputName for key in ems.keys()])
    if not(cutoff is None):
        impMat = np.full((x.shape[0], len(list(z.keys()))), True)
        for i in range(len(list(z.keys()))):
            key = list(z.keys())[i]
            subCols = dImplausibles.loc[:,key]
            if len(np.shape(subCols)) == 1:
                impMat[:,i] = subCols
            else:
                impMat[:,i] = np.apply_along_axis(all, 1, subCols)
    else:
        impMat = np.full((x.shape[0], len(list(z.keys()))), 0, dtype = 'float')
        for i in range(len(list(z.keys()))):
            key = list(z.keys())[i]
            subCols = dImplausibles.loc[:,key]
            if len(np.shape(subCols)) == 1:
                impMat[:,i] = subCols
            else:
                impMat[:,i] = np.apply_along_axis(max, 1, subCols)
    if getRaw:
        return pd.DataFrame(impMat, index = range(impMat.shape[0]), columns = z.keys())
    if not(cutoff is None):
        return np.apply_along_axis(lambda x: sum(x) > len(x) - n, 1, impMat)
    if n == 1:
        imps = np.apply_along_axis(max, 1, impMat)
    else:
        imps = np.apply_along_axis(lambda x: sorted(x, reverse = True)[n-1], 1, impMat)
    imps[imps > maxImp] = maxImp
    return imps
    

# from modeltraining import *

# df = pd.read_csv("../../Desktop/SIRData.csv")
# dfTest = pd.read_csv("../../Desktop/SIRValidation.csv")
# ranges = {'aSI': [0.1, 0.8], 'aIR': [0, 0.5], 'aSR': [0, 0.05]}
# targets = {'nS': [580, 651], 'nI': {'val': 169, 'sigma': 8.45}, 'nR': [199, 221]}
# tefd = emulatorFromData(df, ['nS', 'nI', 'nR'], ranges = ranges, checkRanges = True, verbose = True)
