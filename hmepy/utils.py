import pandas as pd
import numpy as np

__all__ = [
    "evalFuncs", "scaleInput"
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