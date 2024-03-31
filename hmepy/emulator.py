import numbers
import pandas as pd
import numpy as np
from correlations import Correlator

class Emulator:
    """
    Documentation Here.
    """
    def __init__(self, basisF, beta, u, ranges, data = None,
                 model = None, origEm = None, outName = None,
                 aVars = None, discs = None, multiplier = 1):
        self.model = model
        ## Change this line
        self.modelTerms = None
        self.oEm = origEm
        self.bF = basisF
        self.bMu = beta['mu']
        self.bSig = beta['sigma']
        self.uMu = lambda x: 0
        self.multiplier = multiplier
        if isinstance(u['sigma'], numbers.Number):
            self.uSig = self.multiplier * u['sigma']
        else:
            self.uSig = lambda x: self.multiplier * u['sigma'](x)
        self.corr = u['corr']
        if not(outName is None):
            self.outputName = outName
        if ranges is None:
            raise ValueError("No ranges specified for input parameters.")
        self.ranges = ranges
        if aVars is None:
            testDF = pd.DataFrame(np.diag(np.ones(len(ranges))),
                                  columns = ranges.keys())
            tvec = np.zeros((len(self.bF), len(ranges)))
            for i in range(len(self.bF)):
                for j in range(len(ranges)):
                    tvec[i, j] = self.bF[i](testDF.iloc[j,:])
            tvecsum = tvec.sum(axis = 0)
            self.activeVars = tvecsum > 1
        else:
            self.activeVars = aVars
        if not(self.activeVars).all():
            self.activeVars = np.full(len(ranges), True, dtype = bool)
        if not(discs is None):
            try:
                intdisc = discs['internal']
            except:
                intdisc = 0
            try:
                extdisc = discs['external']
            except:
                extdisc = 0
            self.disc = {'internal': intdisc, 'external': extdisc}
        self.betaUCov = lambda x: np.zeros(len(self.bMu))
        if not(data is None):
            ## Do a thing!
            self.inDataPre = data.loc[:,ranges.keys()]
            self.outData = data.loc[:,self.outputName]


tranges = {'a': [0, 1], 'b': [-1, 1], 'c': [2, 3]}
tfuncs = [lambda x: 1, lambda x: x[1], lambda x: x[2], lambda x: x[1]*x[2]]
tbeta = {'mu': np.ones(4), 'sigma': 0}
tu = {'sigma': 2, 'corr': Correlator()}

checking = Emulator(tfuncs, tbeta, tu, tranges)
            