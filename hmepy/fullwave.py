import numpy as np
import pandas as pd

__all__ = ['preflight']

def preflight(data, targets, coff = 0.95, verbose = True, naRm = False):
    if naRm:
        data.dropna(inplace = True)
    potentialProblem = False
    applicableTargets = np.intersect1d(data.columns, list(targets.keys()))
    dAbridge = data.loc[:,applicableTargets]
    targets = {key: targets[key] for key in applicableTargets}
    for key in list(targets.keys()):
        if isinstance(targets[key], dict) and 'val' in targets[key].keys():
            targets[key] = [targets[key]['val'] - 3*targets[key]['sigma'], targets[key]['val'] + 3*targets[key]['sigma']]
    hittingTargets = [
        [dAbridge.loc[ind,tname] >= targets[tname][0] and dAbridge.loc[ind, tname] <= targets[tname][1] for tname in list(targets.keys())]
    for ind in range(np.shape(dAbridge)[0])]
    hittingTargets = pd.DataFrame(hittingTargets, columns = list(targets.keys()))
    for nm in hittingTargets.columns:
        if not any(hittingTargets.loc[:,nm]):
            potentialProblem = True
            if all(dAbridge.loc[:,nm] < targets[nm][0]):
                if verbose: print("Target " + nm + " consistently underestimated.")
            else:
                if verbose: print("Target " + nm + " consistently overestimated.")
            hittingTargets.drop(columns = [nm], inplace = True)
    hittingPoints = [not any(hittingTargets.loc[ind,:]) for ind in range(np.shape(hittingTargets)[0])]
    hittingTargets.drop([ind for ind in range(len(hittingPoints)) if hittingPoints[ind]], inplace = True)
    def checkPairs(df):
        for i in range(np.shape(df)[1]-1):
            for j in range(1, np.shape(df)[1]):
                inm = df.columns[i]
                jnm = df.columns[j]
                cval = np.corrcoef(df.iloc[:,i], df.iloc[:,j])[0,1]
                if cval > coff:
                    if verbose: print("Strong positive correlation between points satisfying " + inm + " and " + jnm)
                if cval < -1*coff:
                    if verbose: print("Strong negative correlation between points satisfying " + inm + " and " + jnm)
    if np.shape(hittingTargets)[0] > 2:
        checkPairs(hittingTargets)
    return potentialProblem