import numpy as np
import pandas as pd
import warnings
from hmepy.modeltraining import *

__all__ = ['getDiagnostic', 'analyzeDiagnostic', 'validationDiagnostics',
           'classificationDiag', 'comparisonDiag', 'standardErrors']

def getDiagnostic(emulator, targets = None, validation = None,
                  whichDiag = 'cd', stDev = 3, cleaned = None,
                  warn = True, kfold = None):
    '''
    Diagnostic Tests for Emulators

    Given an emulator, return a diagnostic measure.

    An emulator's suitability can be checked in a number of ways. This function combines all
    current diagnostics available in the package, returning a context-dependent DataFrame
    containing the results.

    Comparison Diagnostics (cd): Given a set of points, the emulator expectation and variance
    are calculated. This gives a predictive range for the input point according to the
    emulator. We compare this against the actual value given by the simulator: points whose
    emulator prediction is further away from the simulator prediction are to be investigated.
    The 'threshold distance' at which we deem a point too far away is determined by the
    value of stDev, so that a point is worthy of investigation if it lies more than
    stDev*uncertainty away from the simulator prediction.

    Classification Error (ce): Given a set of targets, the emulator can determine implausibility
    of a point with respect to the relevant target, accepting or rejecting as appropriate for a
    given cutoff. We may define a 'simulator' implausibility in a similar fashion to
    that of the emulator, and compare the two implausibility measures. Any point where the
    simulator would not rule out a point but the emulator would should be investigated.

    Standardised Error (se): The known value at a point, combined with the emulator expectation
    and uncertainty, can be used to create a standardised error for the point. This error
    should not be too large, in general, and we would expect an unbiased spread of errors
    around 0. The diagnostic is useful when looking at a collection of validation measures, as
    it can indicate emulators whose other characteristics are worthy of consideration.

    The choice of which diagnostics to use can be controlled by the whichDiag argument;
    the expected form for whichDiag is a list of strings corresponding to the bracketed strings
    above. If performing classification diagnostics, a set of targets must be provided.

    Parameters
    ----------
    emulator: Emulator
        The emulator to test.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        The observations, if desired.
    validation: DataFrame
        The points to validate against.
    whichDiag: [str]
        Which diagnostic measures to apply ('ce', 'cd', 'se').
    stDev: float
        For 'cd', the allowed distance between prediction and reality.
    cleaned: bool | None
        Internal argument for stochastic emulators.
    warn: bool
        Whether a warning is given if 'ce' is chosen without supplied targets.
    kfold: DataFrame | None
        Primarily internal: pre-computed k-fold diagnostics results.

    Returns
    -------
    A DataFrame consisting of the input points, output values, and diagnostics.
    '''

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
    '''
    Diagnostic Analysis for Emulators

    Produces summary statistics for diagnostics

    Given diagnostic information (almost certainly provided from getDiagnostics), we may
    plots the results and highlight the points worthy of concern or further analysis. The output
    depends on the diagnostics desired, which in turn is inferred from the data provided in inData:

    Standardised Error: outliers based on the 'error' column, determined by distance from
    0 error.

    Comparison Diagnostics: points where the emulator and simulator predictions do not overlap;
    that is, those points where the emulator prediction (exp) plus/minus its uncertainty (unc) does not contain
    the simulator output. Where targets are provided, these points are further subsetted: if a simulator
    value is sufficiently far from the observation then the emulator prediction accuracy is not afforded
    any diagnostic weight.

    Classification Error: Points that would be ruled out by the emulator (em) but not by the simulator (sim) are
    returned.

    The inData argument is a DataFrame containing the output values from the simulator, the parameter
    values, and the relevant quantities for the diagnostic tests mentioned above. Where multiple measures
    for multiple tests are included, the points returned are a union of the respective diagnostic checks.

    Parameters
    ----------
    inData: DataFrame
        The diagnostic data.
    outputName: str
        The name of the output being emulated.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        The targets to match to, if relevant.
    plt: bool
        Whether diagnostics should be plotted (currently not implemented).
    cutoff: float
        The cutoff for implausibility measures.
    
    Returns
    -------
    A DataFrame containing points that fail one or more diagnostic tests.
    '''

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
            emInvalid = pd.Series([not p for p in pointInvalid]) & emInvalid
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
            emInvalid = pd.Series([not p for p in pointInvalid]) & emInvalid
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
    '''
    Emulator Diagnostics

    Performs the standard set of validation diagnostics on emulators.

    All the diagnostics here should be performed with a validation (or 'holdout')
    set of data. The presence of a collection of observational targets is optional
    for some of the standard checks but mandatory for others: appropriate warnings
    will be provided in the event that some desired checks cannot be applied.

    The options for diagnostics (with corresponding string identifiers) are:
        
        - Standardised Errors ('se')
        - Comparison Diagnostics ('cd')
        - Classification Errors ('ce')
        - All of the above ('all')

    By default, all diagnostics are performed. For details of each test, see the
    documentation for getDiagnostic.

    Parameters
    ----------
        ems: [Emulator]
            A list of Emulator objects.
        targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
            The list of observations for the outputs of the simulator.
        validation: DataFrame | None
            The validation set, containing input values and simulator outputs.
        analyze: bool
            Should failing points be returned, or the full diagnostic data?
        
    Returns
    -------
    A DataFrame containing either the diagnostic results, or the collection of failed points.

    '''

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
        return [getDiagnostic(ems[i], targets, validation, ad) for i in range(len(ems)) for ad in actualDiag]
    invPoints = np.unique(np.vstack([analyzeDiagnostic(getDiagnostic(ems[i], targets, validation, ad), ems[i].outputName, targets) for i in range(len(ems)) for ad in actualDiag]), axis = 0)
    return pd.DataFrame(invPoints, columns = list(ems[0].ranges.keys()))

def classificationDiag(em, targets, validation, cutoff = 3, plt = False):
    '''
    Classification Diagnostics

    Shorthand for diagnostic test 'ce'.

    Parameters
    ----------
    em: Emulator
        The emulator for the output in question.
    targets: [{'val': float, 'sigma': float}] | [[float, float]]
        The output targets.
    validation: DataFrame | None
        The validation set of points.
    cutoff: float:
        The implausibility cutoff.
    plt: bool
        Whether to produce a plot of the results.
    
    Returns
    -------
    A DataFrame of points that failed the diagnostics.
    '''

    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'ce'), em.outputName, targets, plt, cutoff = cutoff)

def comparisonDiag(em, targets, validation, sd = 3, plt = False):
    '''
    Comparison Diagnostics

    Shorthand for diagnostic test 'cd'.

    Parameters
    ----------
    em: Emulator
        The emulator for the output in question.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        If desired, the observations targets.
    validation: DataFrame | None
        The validation set of points.
    sd: float
        The range of uncertainty allowed for emulator/simulator prediction overlap.
    plt: bool
        Whether or not the results should be plotted.

    Returns
    -------
    A DataFrame of points that failed the diagnostic.
    '''
    
    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'cd', sd), em.outputName, targets, plt)


def standardErrors(em, targets, validation, plt = False):
    '''
    Standard Errors

    Shorthand for the diagnostic test 'se'.

    Parameters
    ----------
    em: Emulator
        The emulator for the output in question.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        If desired, the observational targets.
    validation: DataFrame | None
        The validation set of points.
    plt: bool
        Whether or not the results should be plotted.

    Returns
    -------
    A DataFrame of points failing the diagnostic test.
    '''

    return analyzeDiagnostic(getDiagnostic(em, targets, validation, 'se'), em.outputName, targets, plt)
