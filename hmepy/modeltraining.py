import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.optimize import minimize, Bounds
## Testing
# from correlations import Correlator
# from emulator import Emulator
# from utils import *
## Production
from hmepy.correlations import Correlator
from hmepy.utils import *
from hmepy.emulator import Emulator
import numpy as np
import math
import re
import warnings

__all__ = ['emulatorFromData']

def designMatrix(data, model, paramNames):
    mNames = getFeatureNames(model)
    mFuncs = [nameToFunction(mf, paramNames) for mf in mNames]
    return evalFuncs(mFuncs, data).T

def vcov(data, model, paramNames, outputName):
    X = designMatrix(data, model, paramNames)
    Y = data[outputName]
    Xinv = np.linalg.inv(np.matmul(X.T, X))
    resid = Y - np.matmul(np.matmul(X, Xinv), np.matmul(X.T, Y))
    dof = np.shape(X)[0] - np.shape(X)[1]
    return Xinv * np.sum(resid**2/dof)

def getCoefModel(data, ranges, outputName, add = False,
                  order = 2, verbose = False):
    if verbose: print(outputName)
    Xdat = data.loc[:,ranges.keys()]
    Ydat = data.loc[:,outputName]
    if not(add) and math.comb(len(ranges)+order, len(ranges)) > np.shape(data)[0]:
        if verbose:
            print("""Maximum number of regression terms is greater
                  than the available degress of freedom. Changing to add = True.""")
        add = True
    if add:
        dirStr = "forward"
    else:
        dirStr = "backward"
    model = Pipeline([
        ('poly', PolynomialFeatures(2, include_bias = False)),
        ('SFS', SequentialFeatureSelector(LinearRegression(),
                                          direction = dirStr,
                                          scoring = 'explained_variance',
                                          cv = None)),
        ('linear', LinearRegression())
    ])
    model = model.fit(Xdat, Ydat)
    if explained_variance_score(model.predict(Xdat), Ydat) > 0.9999:
        if verbose:
            print('Model selection suggests perfect fit for output ', outputName, 
                  ' - possible overfit. Removing this output from emulation')
        return None
    return model

def hyperparameterEstimate(inputs, outputs, model, hpRange,
                           corrName = "expSq", beta = None, delta = None,
                           nSteps = 30, verbose = False, logLik = True,
                           performOpt = True):
    if verbose:
        print(outputs.name)
    hpStartDict = {}
    for key in hpRange.keys():
        hpStartDict[key] = hpRange[key][0]
    corr = Correlator(corrName, hpStartDict)
    H = designMatrix(inputs, model, list(inputs.columns))
    avTest = np.diag(np.ones(np.shape(inputs)[1]))
    avH = designMatrix(pd.DataFrame(data = avTest, columns = list(inputs.columns)), model, list(inputs.columns))
    isActive = [sum(avH[i,:]) > 1 for i in range(np.shape(avH)[0])]
    if sum(isActive) == 0:
        isActive = [True for i in range(len(isActive))]
    def corrMat(points, hp, delt):
        thisCorr = corr.setHyperp(hp, delt)
        return thisCorr.getCorr(points, actives = isActive)
    def funcToOpt(params, logLik = logLik, returnStats = False):
        hp = dict(zip(hpRange.keys(), params[0:(len(params))]))
        if corrName == "matern":
            if (hp['nu'] > 0.5 and hp['nu'] <= 1): hp['nu'] = 0.5
            if (hp['nu'] > 1 and hp['nu'] <= 2): hp['nu'] = 1.5
            if (hp['nu'] > 2): hp['nu'] = 2.5
        try:
            delta = params[-1]
        except:
            delta = 0
        if delta is None:
            delta = 0
        A = corrMat(inputs, hp, delta)
        aDet = np.linalg.det(A)
        if aDet < 0:
            aDet = 1e-20
        Ainv = np.linalg.inv(A)
        if beta is None:
            invMat = np.linalg.inv(np.matmul(np.matmul(H.T, Ainv), H))
            bML = np.matmul(np.matmul(invMat, H.T), np.matmul(Ainv, outputs))
        else:
            bML = beta
        modDiff = outputs - np.matmul(H, bML)
        sigSqML = np.matmul(modDiff.T, np.matmul(Ainv, modDiff))/len(outputs)
        if logLik:
            lik = -len(outputs) * math.log(sigSqML)/2 - math.log(aDet)/2
        else:
            lik = 1/math.sqrt(sigSqML**len(outputs)) * 1/math.sqrt(aDet)
        if math.isinf(lik):
            lik = -math.inf
        if returnStats:
            return {'beta': bML, 'sigma': float(math.sqrt(sigSqML))}
        return lik
    ## Possibly some gradient stuff here.
    if hasattr(hpRange, 'theta') and len(hpRange['theta']) == 1:
        if delta is None:
            delta = 0.05
        bestPoint = hpRange
        bestDelta = delta
        bestParams = bestPoint.append(delta)
    else:
        if corrName == "matern":
            gridSearch = np.array([(x,y) for x in np.arange(hpRange['theta'][0], hpRange['theta'][1], np.diff(hpRange['theta'])/(nSteps*4)) for y in [0.5, 1.5, 2.5, 3.5]])
        else:
            if len(hpRange) == 1:
                gridSearch = np.array(np.arange(list(hpRange.values())[0][0], list(hpRange.values())[0][1], np.diff(list(hpRange.values())[0])/(nSteps*4)))
            if len(hpRange) == 2:
                firstRange = np.arange(list(hpRange.values())[0][0], list(hpRange.values())[0][1], np.diff(list(hpRange.values())[0])/nSteps)
                secondRange = np.arange(list(hpRange.values())[1][0], list(hpRange.values())[1][1], np.diff(list(hpRange.values())[1])/nSteps)
                gridSearch = np.array([(x,y) for x in firstRange for y in secondRange])
        if delta is None:
            dval = 0.01
        else:
            dval = delta
        if len(np.shape(gridSearch)) == 1:
            gridLiks = [funcToOpt([x, delta]) for x in gridSearch]
        else:
            gridLiks = np.apply_along_axis(lambda x: funcToOpt(np.append(x,delta)), 1, gridSearch)
        bestLik = np.argmax(gridLiks)
        if len(np.shape(gridSearch)) == 1:
            bestPoint = {list(hpRange.keys())[0]: gridSearch[bestLik]}
        else:
            bestPoint = dict(zip(hpRange.keys(), gridSearch[bestLik,:]))
        if sum(isActive) == np.shape(inputs)[1]:
            delta = 0
        if delta is None:
            delta = 0.05
        if delta == 0:
            bestDelta = 1e-10
        else:
            bestDelta = delta
        if performOpt:
            initialParams = np.append(list(bestPoint.values()), bestDelta)
            lower = np.append([hp[0] for hp in list(hpRange.values())], 0)
            upper = np.append([hp[1] for hp in list(hpRange.values())], 0.5)
            bds = Bounds(lower, upper)
            optimed = minimize(lambda x: -1*funcToOpt(x), initialParams, bounds = bds)
            bestParams = optimed.x
        else:
            bestParams = np.append(list(bestPoint.values()), bestDelta)
        otherPars = funcToOpt(bestParams, returnStats = True)
        allPoints = {'hp': dict(zip(hpRange.keys(), bestParams[0:len(bestParams)])),
                     'delta': bestParams[-1], 'sigma': otherPars['sigma'],
                     'beta': otherPars['beta']}
    return allPoints

def emulatorFromData(inputData, outputNames, ranges = None,
                     inputNames = None, emulatorType = None,
                     specifiedPriors = None, order = 2, betaVar = False,
                     corrName = "expSq", adjusted = True, discrepancies = None,
                     verbose = False, naRm = False, checkRanges = True,
                     targets = None, hasHierarchy = False, covOpts = None,
                     *args):
    if not(all([outname in inputData.columns for outname in outputNames])):
        raise ValueError("outputNames do not match data.")
    if not(targets is None):
        ## Do some preflight stuff?
        pass
    if ranges is None:
        if inputNames is None:
            raise ValueError("One of ranges or inputNames must be provided.")
        else:
            warnings.warn("No ranges provided: inputs assumed to be in range [-1, 1] for all parameters.")
            ranges = {}
            for name in inputNames:
                ranges[name] = [-1,1]
    if any([np.diff(ran)<=0 for ran in ranges.values()]):
        raise ValueError("Ranges either not specified, or misspecified.")
    if emulatorType is None:
        emulatorType = "default"
    ## Stuff about variance emulation and repetitions at points
    if naRm:
        inputData = inputData.dropna()
    if checkRanges:
        newRanges = {}
        for name in ranges.keys():
            coldat = inputData.loc[:,name]
            maxmin = ranges[name]
            coldiff = np.ptp(coldat)
            lowBd = max(maxmin[0], np.min(coldat) - 0.05*coldiff)
            upBd = min(maxmin[1], np.max(coldat) + 0.05*coldiff)
            newRanges[name] = [lowBd, upBd]
        ranges = newRanges
    if np.shape(inputData)[0] < 10*len(ranges):
        warnings.warn("Fewer than " + 10*len(ranges) + " valid points in " + len(ranges) + " dimensions - treat emulated outputs with caution or include more data points (minimum 10 times number of input parameters).")
    inputScaled = evalFuncs(scaleInput, inputData.loc[:,ranges.keys()], ranges)
    for name in outputNames:
        inputScaled[name] = inputData[name]
    if emulatorType == "default":
        if not(hasattr(specifiedPriors, "func")):
            if verbose:
                print("Fitting regression surfaces...")
            models = [getCoefModel(inputScaled, ranges, oName, order = order, verbose = verbose) for oName in outputNames]
            isValidModel = [not(mod is None) for mod in models]
            outputNames = [o for (o,v) in zip(outputNames, isValidModel) if v]
            models = [m for (m, v) in zip(models, isValidModel) if v]
            if len(models) == 0:
                raise ArithmeticError("No outputs can be robustly represented by regression surfaces at this order.")
            modelBetaMus = [np.insert(m.named_steps['linear'].coef_, 0, m.named_steps['linear'].intercept_) for m in models]
            modelFeatNames = [getFeatureNames(m) for m in models]
            modelBasisF = [[nameToFunction(mfi, list(ranges.keys())) for mfi in mf] for mf in modelFeatNames]
            if not(betaVar):
                modelBetaSigmas = [np.zeros((len(mbm), len(mbm))) for mbm in modelBetaMus]
            else:
                modelBetaSigmas = [vcov(inputScaled, models[ind], list(ranges.keys()), outputNames[ind]) for ind in range(len(models))]
        else:
            ## If priors are provided, they're allocated here.
            pass
        if verbose:
            print("Building correlation structures...")
        if not(specifiedPriors is None) and hasattr(specifiedPriors, 'hyperP'):
            ## Allocate priors
            pass
        else:
            ## Want to catch ill-defined correlation names here.
            # For now, assume all is well.
            if not corrName in ['expSq', 'ornUhl', 'matern', 'ratQuad']:
                try:
                    thran = [thetaRanges for i in range(len(models))]
                except:
                    warnings.warn("User-defined correlation function " + corrName + " chosen but no corresponding hyperparameter ranges. Reverting to expSq.")
                    corrName = 'expSq'
            if corrName == 'expSq' or corrName == 'ornUhl':
                thran = [{'theta': [min(2/(order+1), 1/3), 2/order]} for i in range(len(models))]
            if corrName == 'matern':
                thran = [{'theta': [min(2/(order+1), 1/3), 2/order], 'nu': [0.5, 3.5]} for i in range(len(models))]
            if corrName == 'ratQuad':
                thran = [{'theta': [min(2/(order+1), 1/3), 2/order], 'alpha': [0.01,5]} for i in range(len(models))]
        specs = [hyperparameterEstimate(inputScaled.loc[:,ranges.keys()],
                                        inputScaled.loc[:,outputNames[i]],
                                        models[i],
                                        thran[i],
                                        corrName,
                                        modelBetaMus[i],
                                        delta = None,
                                        verbose = verbose) for i in range(len(models))]
        if verbose:
            print("Creating emulators...")
        emlist = {}
        for i in range(len(models)):
            newEm = Emulator(modelBasisF[i], beta = {'mu': modelBetaMus[i], 'sigma': modelBetaSigmas[i]},
                             u = {'sigma': specs[i]['sigma'], 'corr': Correlator(corrName, specs[i]['hp'], specs[i]['delta'])},
                             ranges = ranges, model = models[i], outName = outputNames[i])
            emlist[outputNames[i]] = newEm
        if (adjusted):
            for i in outputNames:
                emlist[i] = emlist[i].adjust(inputData, emlist[i].outputName)
        return emlist

# df = pd.read_csv("../../Desktop/SIRData.csv")
# dfTest = pd.read_csv("../../Desktop/SIRValidation.csv")
# ranges = {'aSI': [0.1, 0.8], 'aIR': [0, 0.5], 'aSR': [0, 0.05]}
# tefd = emulatorFromData(df, ['nS', 'nI', 'nR'], ranges = ranges, checkRanges = True, verbose = True)
# print(tefd['nI'])
# preds = tefd['nI'].getExp(dfTest)
# print(preds - dfTest['nI'])
# print(tefd['nI'].getCov(dfTest, full = True))