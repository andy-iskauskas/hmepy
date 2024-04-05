import pandas as pd
import numpy as np
import re

__all__ = [
    "evalFuncs", "scaleInput", "nameToFunction",
    "getFeatureNames"
]

def evalFuncs(funcs, points, *args):
    isCall = not callable(funcs)
    isMany = len(np.shape(points)) > 1
    if isCall and isMany:
        return np.vstack([points.apply(funcs[i], args = args, axis = 1) for i in range(len(funcs))])
    if isCall:
        return [funcs[i](points, *args) for i in range(len(funcs))]
    if isMany:
        return(points.apply(funcs, args = args, axis = 1))
    return funcs(points, *args)

def scaleInput(points, ranges, forward = True):
    centres = [np.sum(ranges[key])/2 for key in ranges.keys()]
    scales = [np.diff(ranges[key])[0]/2 for key in ranges.keys()]
    if forward:
        if len(np.shape(points)) == 1:
            return (points - centres)/scales
        return points.apply(lambda row: (row-centres)/scales)
    if len(np.shape(points)) == 1:
        return points * scales / centres
    return points.apply(lambda row: row * scales + centres)

def getFeatureNames(model):
    chosenPars = model.named_steps['SFS'].get_feature_names_out()
    originalPars = model.named_steps['poly'].get_feature_names_out()
    return np.insert(originalPars[[int(re.sub("x", "", par)) for par in chosenPars]], 0, "1")

def nameToFunction(funcName, varNames):
    if str == "1":
        return lambda x: 1
    strPow = re.sub("\^(\d+)", "**\\1", funcName)
    strTimes = re.sub("\s", "*", strPow)
    varNamesOrdered = sorted(varNames, key = len)
    for vn in varNamesOrdered:
        varind = varNames.index(vn)
        strTimes = re.sub(vn, "\u00a3\u00a3\u00a3\u00a3[" + str(varind) + "]", strTimes)
    strTimes = re.sub("\u00a3\u00a3\u00a3\u00a3", "x", strTimes)
    return eval("lambda x: " + strTimes)
