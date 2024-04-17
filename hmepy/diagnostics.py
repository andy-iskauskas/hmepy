import numpy as np
import pandas as pd
import warnings
## Testing
# from modeltraining import *
## Production
from hmepy.modeltraining import *

__all__ = ['getDiagnostic', 'analyzeDiagnostic', 'validationDiagnostics',
           'classificationDiag', 'comparisonDiag', 'standardErrors']

def getDiagnostic(emulator, targets = None, validation = None,
                  whichDiag = 'cd', stDev = 3, cleaned = None,
                  warn = True, kfold = None):
    if targets is None and whichDiag == 'ce':
        raise ValueError("Targets must be provided for classification error diagnostics.")
    if validation is None:
        ## Need to apply k-fold validation here
        raise ValueError("Validation set is currently needed for any diagnostic.")
    inputPoints = validation.loc[:,emulator.ranges.keys()]
    outputPoints = validation.loc[:,emulator.outputName]
    if whichDiag == 'ce':
        emImp = emulator.implausibility(inputPoints, targets[emulator.outputName])
    else:
        emExp = emulator.getExp(inputPoints)
        emCov = emulator.getCov(inputPoints)
    if whichDiag == 'se':
        num = emExp - outputPoints
        denom = np.sqrt(emCov + emulator.disc['internal']**2 + emulator.disc['external']**2)
        errors = num/denom
        outData = np.hstack([validation.loc[:,np.append(list(emulator.ranges.keys()), emulator.outputName)], np.reshape(errors, (len(errors), 1))])
        outData = pd.DataFrame(outData, columns = np.append(list(emulator.ranges.keys()), [emulator.outputName, 'error']))
    if whichDiag == 'cd':
        if stDev is None: stDev = 3
        emUnc = stDev * np.sqrt(emCov + emulator.disc['internal']**2 + emulator.disc['external']**2)
        outData = np.hstack([validation.loc[:,np.append(list(emulator.ranges.keys()), emulator.outputName)], np.reshape(emExp, (len(emExp), 1)), np.reshape(emUnc, (len(emUnc), 1))])
        outData = pd.DataFrame(outData, columns = np.append(list(emulator.ranges.keys()), [emulator.outputName, 'exp', 'unc']))
    if whichDiag == 'ce':
        thisTarget = targets[emulator.outputName]
        if not isinstance(thisTarget, dict):
            simImp = np.abs(outputPoints - np.mean(thisTarget))/(np.diff(thisTarget)/2)
        else:
            simImp = np.sqrt((outputPoints - thisTarget['val'])**2)/thisTarget['sigma']
        outData = np.hstack([validation.loc[:,np.append(list(emulator.ranges.keys()), emulator.outputName)], np.reshape(emImp, (len(emImp), 1)), np.reshape(simImp, (len(simImp), 1))])
        outData = pd.DataFrame(outData, columns = np.append(list(emulator.ranges.keys()), [emulator.outputName, 'em', 'sim']))
    return outData

def analyzeDiagnostic(inData, outputName, targets = None,
                      plt = False, cutoff = 3):
    outputPoints = inData[outputName]
    inColNames = list(inData.columns)
    isInput = [not elem in [outputName, 'error', 'exp', 'unc', 'em', 'sim'] for elem in list(inData.columns)]
    inputNames = [i for (i, v) in zip(inColNames, isInput) if v]
    inputPoints = inData.loc[:,inputNames]
    if 'error' in inColNames:
        if plt:
            pass
        emInvalid = np.abs(inData['error'] > 3)
        if not(targets is None):
            thisTarget = targets[outputName]
            if not isinstance(thisTarget, dict):
                pointInvalid = (outputPoints < np.array(len(outputPoints)).fill(thisTarget[0]-np.diff(thisTarget))) | (outputPoints > np.array(len(outputPoints)).fill(thisTarget[1]+np.diff(thisTarget)))
            else:
                pointInvalid = (outputPoints < np.array(len(outputPoints)).fill(thisTarget['val'] - 6*thisTarget['sigma'])) | (outputPoints > np.array(len(outputPoints)).fill(thisTarget['val'] + 6*thisTarget['sigma']))
            if plt:
                pass
            emInvalid = [not p for p in pointInvalid] & emInvalid
    if 'exp' in inColNames:
        emExtent = np.append(inData['exp']+inData['unc'], inData['exp']-inData['unc'])
        emRanges = np.max(emExtent)-np.min(emExtent)
        emInvalid = (outputPoints >  inData['exp'] + inData['unc']) | (outputPoints < inData['exp'] - inData['unc'])
        if not(targets is None):
            thisTarget = targets[outputName]
            if not isinstance(thisTarget, dict):
                pointInvalid = (outputPoints < np.array(len(outputPoints)).fill(thisTarget[0]-np.diff(thisTarget))) | (outputPoints > np.array(len(outputPoints)).fill(thisTarget[1]+np.diff(thisTarget)))
                panLims = [thisTarget[0] - np.diff(thisTarget)/4, thisTarget[1] + np.diff(thisTarget)/4]
            else:
                pointInvalid = (outputPoints < np.array(len(outputPoints)).fill(thisTarget['val'] - 6*thisTarget['sigma'])) | (outputPoints > np.array(len(outputPoints)).fill(thisTarget['val'] + 6*thisTarget['sigma']))
                panLims = [thisTarget['val'] - 4.5*thisTarget['sigma'], thisTarget['val'] + 4.5*thisTarget['sigma']]
            emInvalid = [not p for p in pointInvalid] & emInvalid
        else:
            panLims = None
        if plt:
            pass
    if 'em' in inColNames:
        if targets is None:
            raise KeyError("Require target to check classification error.")
        if cutoff is None:
            cutoff = 3
        thisTarget = targets[outputName]
        if not isinstance(thisTarget, dict):
            tCut = cutoff/3
        else:
            tCut = cutoff
        emInvalid = (inData['em'] > cutoff) & (inData['sim'] <= tCut)
        if plt:
            pass
    return inputPoints.loc[emInvalid,:]

def validationDiagnostics(ems, targets = None, validation = None,
                          whichDiag = ['cd', 'ce', 'se'], analyze = True):
    if whichDiag == 'all' or (isinstance(whichDiag, (list, pd.core.series.Series, np.ndarray)) and 'all' in whichDiag):
        whichDiag = ['cd', 'ce', 'se']
    isDiag = [w in ['cd', 'ce', 'se'] for w in whichDiag]
    actualDiag = [i for (i, v) in zip(whichDiag, isDiag) if v]
    if not len(whichDiag) == len(actualDiag):
        nonDiag = [not w in ['cd', 'ce', 'se'] for w in whichDiag]
        warnings.warn("Unrecognised diagnostics: " + ", ".join(nonDiag) + "\n\tValid diagnostic labels are 'cd', 'se', 'ce' or 'all'.")
    actualDiag = np.unique(actualDiag)
    if 'ce' in actualDiag and targets is None:
        warnings.warn("No targets provided; cannot perform classification diagnostics")
        actualDiag = [w for w in actualDiag if not(w == 'ce')]
    if not analyze:
        return [getDiagnostic(ems[oname], targets, validation, ad) for oname in list(ems.keys()) for ad in actualDiag]
    invPoints = np.unique(np.vstack([analyzeDiagnostic(getDiagnostic(ems[oname], targets, validation, ad), ems[oname].outputName, targets) for oname in ems.keys() for ad in actualDiag]), axis = 0)
    return pd.DataFrame(invPoints, columns = list(ems[list(ems.keys())[0]].ranges.keys()))

def classificationDiag(em, targets, validation, cutoff = 3, plt = False):
    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'ce'), em.outputName, targets, plt, cutoff = cutoff)
def comparisonDiag(em, targets, validation, sd = 3, plt = False):
    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'cd', sd), em.outputName, targets, plt)
def standardErrors(em, targets, validation, plt = False):
    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'se'), em.outputName, targets, plt)


# df = pd.read_csv("../../Desktop/SIRData.csv")
# dfTest = pd.read_csv("../../Desktop/SIRValidation.csv")
# ranges = {'aSI': [0.1, 0.8], 'aIR': [0, 0.5], 'aSR': [0, 0.05]}
# targets = {'nS': [580, 651], 'nI': {'val': 169, 'sigma': 8.45}, 'nR': [199, 221]}
# tefd = emulatorFromData(df, ['nS', 'nI', 'nR'], ranges = ranges, checkRanges = True, verbose = True)
# testem = tefd['nI']

# print(validationDiagnostics(tefd, targets, dfTest))