import numbers
import pandas as pd
import numpy as np
from hmepy.correlations import Correlator
from hmepy.utils import *

__all__ = ["Emulator"]

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
        self.betaUCov = [lambda x: 0 for i in range(len(self.bMu))]
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
        if not(data is None):
            self.inData = evalFuncs(scaleInput, data.loc[:,ranges.keys()], self.ranges)
            self.outData = data.loc[:,self.outputName]
        if hasattr(self, 'inData'):
            if isinstance(self.uSig, numbers.Number):
                dCorr = self.uSig**2 * self.corr.getCorr(self.inData, actives = self.activeVars)
            else:
                sCorr = self.corr.getCorr(self.inData, actives = self.activeVars)
                dCorr = np.vstack([sCorr[i,:] * evalFuncs(self.uSig, self.inData)**2 for i in len(np.shape(sCorr)[0])])
            self.dCorrs = np.linalg.inv(dCorr)
            self.designMat = evalFuncs(self.bF, self.inData).T
            self.uVarMod = np.matmul(np.matmul(
                np.matmul(self.dCorrs, self.designMat), 
                np.matmul(self.bSig, self.designMat.T)),
                self.dCorrs)
            self.uExpMod = np.matmul(self.dCorrs,
                                     self.outData - np.matmul(self.designMat, self.bMu))
            self.betaUCovMod = np.matmul(
                np.matmul(self.bSig, self.designMat.T),
                self.dCorrs
            )
    
    def getExp(self, x, includeC = True, cData = None):
        x = evalFuncs(scaleInput, x.loc[:,self.ranges.keys()], self.ranges)
        if np.count_nonzero(self.bSig) != 0 or self.model is None:
            g = evalFuncs(self.bF, x).T
            betaPart = [np.matmul(g[i], self.bMu) for i in range(np.shape(g)[0])]
        else:
            ## Something something predict
            pass
        bU = evalFuncs(self.betaUCov, x)
        uPart = x.apply(self.uMu, axis = 1)
        if hasattr(self, 'inData'):
            if cData is None:
                cData = self.corr.getCorr(self.inData, x, self.activeVars)
            if isinstance(self.uSig, numbers.Number):
                added = np.dot(
                    np.matmul(bU.T, self.designMat.T) + self.uSig**2 * cData,
                    self.uExpMod)
                uPart = uPart + added
            else:
                sigPreds = np.vstack([cData[i,:] * evalFuncs(self.uSig, x)**2 for i in range(len(np.shape(cData)[0]))])
                uPart = uPart + np.matmul(
                    np.matmul(bU.T, self.designMat.T) + sigPreds,
                    self.uExpMod
                )
        if includeC:
            return betaPart + uPart     
        return betaPart
    def getCov(self, x, xp = None, full = False, includeC = True, cX = None, cXP = None):
        betaPart = np.zeros(np.shape(x)[0])
        x = evalFuncs(scaleInput, x.loc[:,self.ranges.keys()], self.ranges)
        if np.count_nonzero(self.bSig) != 0:
            gX = evalFuncs(self.bF, x).T
        else:
            gX = None
        bUX = evalFuncs(self.betaUCov, x)
        noneFlag = False
        if xp is None:
            noneFlag = True
            xp = x
            gXP = gX
            bUXP = bUX
        else:
            xp = evalFuncs(scaleInput, xp.loc[:,self.ranges.keys()], self.ranges)
            if np.count_nonzero(self.bSig) != 0:
                gXP = evalFuncs(self.bF, xp).T
            else:
                gXP = None
            bUXP = evalFuncs(self.betaUCov, xp)
        if full or np.shape(x)[0] != np.shape(xp)[0]:
            xXpC = self.corr.getCorr(xp, x, self.activeVars)
            if not(gX is None):
                betaPart = np.matmul(np.matmul(gX, self.bSig), gXP.T)
            if isinstance(self.uSig, numbers.Number):
                uPart = self.uSig**2 * xXpC
            else:
                uPart = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, xp)) * xXpC
            if hasattr(self, 'inData'):
                if cX is None:
                    cX = self.corr.getCorr(self.inData, x, self.activeVars)
                if cXP is None:
                    if noneFlag:
                        cXP = cX
                    else:
                        cXP = self.corr.getCorr(self.inData, xp, self.activeVars)
                if isinstance(self.uSig, numbers.Number):
                    uPart = uPart - self.uSig**4 * np.matmul(
                        np.matmul(cX, self.dCorrs - self.uVarMod),
                        cXP.T
                    )
                    if np.count_nonzero(self.betaUCovMod) != 0:
                        bUX = bUX - np.matmul(self.betaUCovMod,
                                              np.matmul(self.designMat, bUX) + self.uSig**2 * cX.T)
                        if noneFlag:
                            bUXP = bUX
                        else:
                            bUXP = bUXP - np.matmul(self.betaUCovMod,
                                                    np.matmul(self.designMat, bUXP) + self.uSig**2 * cXP.T)
                else:
                    cX = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, self.inData)) * cX
                    cXP = np.outer(evalFuncs(self.uSig, xp), evalFuncs(self.uSig, self.inData)) * cXP
                    uPart = uPart - np.matmul(cX,
                                              np.matmul(self.dCorrs - self.uVarMod), cXP.T)
                    if np.count_nonzero(self.betaUCovMod) != 0:
                        bUX = bUX - np.matmul(self.betaUCovMod,
                                              np.matmul(self.designMat, bUX) + cX.T)
                        bUXP = bUXP - np.matmul(self.betaUCov,
                                                np.matmul(self.designMat, bUXP) + cXP.T)
            if np.count_nonzero(bUX) != 0:
                bU = np.matmul(gX, bUXP) + np.matmul(bUX.T, gXP.T) 
            else:
                bU = np.zeros(np.shape(x)[0])
        else:
            ## Here will be the "quicker" method for when we only want the diagonal.
            indexAlong = range(np.shape(x)[0])
            if not(gX is None):
                betaPart = [np.matmul(gX.iloc[:,i], np.matmul(self.bSig, gXP.iloc[:,i])) for i in indexAlong]
            if np.array_equal(x, xp):
                if isinstance(self.uSig, numbers.Number):
                    uPart = [self.uSig**2 for i in indexAlong]
                else:
                    uPart = evalFuncs(self.uSig, x)**2
            else:
                if isinstance(self.uSig, numbers.Number):
                    uPartS = self.uSig**2
                else:
                    uPartS = np.diag([self.uSig(x.loc[i,:])*self.uSig(xp.loc[i,:]) for i in indexAlong])
                uPart = uPartS * np.diag(self.corr.getCorr(x, xp, self.activeVars))
            if hasattr(self, 'inData'):
                if cX is None:
                    cX = self.corr.getCorr(self.inData, x, self.activeVars)
                if cXP is None:
                    if noneFlag:
                        cXP = cX
                    else:
                        cXP = self.corr.getCorr(self.inData, xp, self.activeVars)
                if isinstance(self.uSig, numbers.Number):
                    cX = cX * self.uSig**2
                    cXP = cXP * self.uSig**2
                else:
                    cX = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, self.inData)) * cX
                    cXP = np.outer(evalFuncs(self.uSig, xp), evalFuncs(self.uSig, self.inData)) * cXP
                uPart = uPart - np.sum(np.matmul(cX, self.dCorrs - self.uVarMod) * cXP, axis = 1)
                if np.count_nonzero(self.betaUCovMod) != 0:
                    bUX = bUX - np.matmul(self.betaUCovMod,
                                          np.matmul(self.designMat, bUX) + cX.T)
                    if noneFlag:
                        bUXP = bUX
                    else:
                        bUXP = bUXP - np.matmul(self.betaUCovMod,
                                                np.matmul(self.designMat, bUXP) + cXP.T)
            if np.count_nonzero(bUX) != 0:
                gValX = [evalFuncs(self.bF, x.iloc[[i],:]) for i in indexAlong]
                gValXP = [evalFuncs(self.bF, x.iloc[[i],:]) for i in indexAlong]
                bU = [np.dot(gValX.iloc[i,:], bUXP.iloc[:,i]) + np.dot(gValXP.iloc[i,:], bUX.iloc[:,i]) for i in indexAlong]
            else:
                bU = np.zeros(np.shape(x)[0])
        if includeC:
            result = betaPart + uPart + bU
        else:
            result = betaPart
        return result
    def __str__(self):
        outString = "Parameters and Ranges\n"
        for key in self.ranges.keys():
            outString = outString + key + ": (" + ", ".join(str(ran) for ran in self.ranges[key]) + "); "
        outString = outString[:-2] + "\n"
        outString = outString + "Specifications:\n"
        outString = outString + "\tBasis Functions: "
        if not(self.model is None):
            pass
            # Do a thing
        else:
            # Do another thing (maybe need to add function_to_names into utils)
            outString = outString + "Coming soon\n"
        outString = outString + "\tActive Variables: "
        outString = outString + ", ".join([i for (i, v) in zip(list(self.ranges.keys()), self.activeVars) if v]) + "\n"
        outString = outString + "\tRegression Surface Expectation: "
        outString = outString + "; ".join(str(b) for b in self.bMu) + "\n"
        outString = outString + "\tRegression Surface Variance (Eigenvalues): "
        outString = outString + "(" + "; ".join(str(ev) for ev in np.linalg.eigvals(self.bSig)) + ")\n"
        outString = outString + "Correlation Structure:\n"
        if hasattr(self, 'dCorrs'):
            outString = outString + "Bayes-adjusted emulator - prior specifications listed.\n"
        outString = outString + "\tPrior Variance: "
        if isinstance(self.uSig, numbers.Number):
            outString = outString + str(self.uSig**2) + "\n"
        else:
            outString = outString + str(self.uSig([sum(self.ranges[key])/2 for key in self.ranges.keys()])) + "\n"
        outString = outString + "\tExpectation: " + str(self.uMu(np.zeros(len(self.ranges)))) + "\n"
        outString = outString + self.corr.__str__(prepend = "\t")
        outString = outString + "Mixed Covariance: "
        outString = outString + "(" + ", ".join([str(bc) for bc in evalFuncs(self.betaUCov, np.zeros(len(self.ranges)))]) + ")\n"
        return f"{outString}"