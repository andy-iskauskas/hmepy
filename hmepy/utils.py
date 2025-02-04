import pandas as pd
import numpy as np
import re

__all__ = [
    "evalFuncs", "scaleInput", "nameToFunction",
    "getFeatureNames", "prinVar"
]

## Helper to evaluate function(s) across point(s)
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

## Scales inputs based on the ranges, either from the ranges to [-1, 1] or from [-1, 1] to the ranges
def scaleInput(points, ranges, forward = True):
    centres = [np.sum(ranges[key])/2 for key in ranges.keys()]
    scales = [np.diff(ranges[key])[0]/2 for key in ranges.keys()]
    if forward:
        if len(np.shape(points)) == 1:
            return (points - centres)/scales
        return points.apply(lambda row: (row-centres)/scales)
    if len(np.shape(points)) == 1:
        return points * scales + centres
    return points.apply(lambda row: row * scales + centres)

## Obtains the parameter names from a LinearRegression model
def getFeatureNames(model):
    chosenPars = model.named_steps['SFS'].get_feature_names_out()
    originalPars = model.named_steps['poly'].get_feature_names_out()
    return np.insert(originalPars[[int(re.sub("x", "", par)) for par in chosenPars]], 0, "1")

## Converts named regression terms to functions of a vector
def nameToFunction(funcName, varNames):
    if str == "1":
        return lambda x: 1
    strPow = re.sub("\\^(\\d+)", "**\\1", funcName)
    strTimes = re.sub("\\s", "*", strPow)
    varNamesOrdered = sorted(varNames, key = len)
    for vn in varNamesOrdered:
        varind = varNames.index(vn)
        strTimes = re.sub(vn, "\u00a3\u00a3\u00a3\u00a3[" + str(varind) + "]", strTimes)
    strTimes = re.sub("\u00a3\u00a3\u00a3\u00a3", "x", strTimes)
    return eval("lambda x: " + strTimes)

def prinVar(data, max_vars = None, var_cut = None):
    if (max_vars is None):
        max_vars = data.shape[1]
    selected_vars = []
    agg_r2 = []
    tr = []
    cum_var = []
    num_vars = data.shape[1]
    if ((not var_cut is None) and isinstance(var_cut, float)):
        if (var_cut < 0 or var_cut > 1):
            var_cut = 1
    cond_cor = np.array(data.corr())
    all_vars = list(data.columns)
    for n in range(max_vars):
        ar2 = [sum(i*i for i in cond_cor[j,:]) for j in range(cond_cor.shape[0])]
        max_ar2_val = max(ar2)
        max_ar2_ind = ar2.index(max_ar2_val)
        agg_r2.append(max_ar2_val)
        selected_vars.append(all_vars[max_ar2_ind])
        all_vars.remove(all_vars[max_ar2_ind])
        S22 = np.delete(cond_cor, max_ar2_ind, 0)
        S22 = np.delete(S22, max_ar2_ind, 1)
        S21 = np.delete(cond_cor, max_ar2_ind, 0)
        S21 = S21[:,max_ar2_ind]
        cond_cor = S22 - [[i*j for i in S21] for j in S21]/cond_cor[max_ar2_ind, max_ar2_ind]
        new_tr = sum([cond_cor[i,i] for i in range(np.shape(cond_cor)[0])])
        tr.append(new_tr)
        cum_var.append(1-new_tr/num_vars)
        if ((not var_cut is None) and isinstance(var_cut, float)):
            if (cum_var[len(cum_var)-1] >= var_cut):
                break
    return {"ordered_variables": selected_vars,
    "aggregate_r2": agg_r2,
    "traces": tr,
    "cumulative_variance": cum_var}
