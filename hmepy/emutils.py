import numpy as np
import pandas as pd
from hmepy.emulator import Emulator
from hmepy.modeltraining import emulatorFromData
from hmepy.implausibility import nthImplausible

__all__ = ['collectEmulators', 'getRanges']

'''
    Emulator Collation

    Collects and orders emulators for use in other functions.

    Most often used as a pre-processing stage for functions such as nthImplausible
    and generateNewDesign, this function unnests lists of lists of emulators (a commmon
    output after multiple waves of emulation), ordering them in terms of their possible
    restrictiveness for efficiency in calculating implausbility.

    Criteria for ordering and the short form string are as follows:
    - Number of Parameters ('p'): Emulators with more parameters are likely to have
      arisen from later-wave emulation, so take precedence;
    - Implausibility ('i', requires targets): A test collection of points is created,
      with size sampleSize, and the emulator implausibilities are calculated for each
      emulator. The most restrictive ones are placed first.
    - Input space volume ('v'): A hyperparameter volume is created for each emulator
      using em.ranges; those with smaller volume are likely to have come from later
      waves of emulation, and take precedence.
    The criteria can be ordered in whichever fashion is desired: the default is that
    number of parameters is considered first, using implausibility (if targets are
    provided) to break ties, and using hyperrectangle volume to break any remaining ties.
    If two emulators are indistinguishable using the metrics, they maintain the order
    possesses prior to the collectEmulators call.

    Parameters:
    -----------
    ems: [Emulator] | [[Emulator], [Emulator], ...]
        The list (or nested list) or Emulator objects.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        Targets for the outputs (if 'i' is used).
    cutoff: float
        If using option 'i', the cutoff for implausibility.
    ordering: [str]
        The metrics to use, and the order in which to perform them.
    sampleSize: int
        How many points to consider implausibility for if using option 'i'.
    
    Returns:
    --------
    A flattened list of emulators, ordered with respect to the chosen metrics.
'''
def collectEmulators(ems, targets = None, cutoff = 3,
                     ordering = ['p', 'i', 'v'], sampleSize = 200):
    if isinstance(ems, Emulator):
        return {ems.outputName: ems}
    if all(isinstance(em, Emulator) for em in ems):
        emNames = [em.outputName for em in ems]
        emRangeLength = [len(em.ranges) for em in ems]
        emRangeProds = [np.prod([np.diff(ran) for ran in em.ranges.values()]) for em in ems]
        if targets is None:
            whichOrder = list(set(['p', 'v']).intersection(ordering))
        else:
            whichOrder = list(set(['p', 'i', 'v']).intersection(ordering))
        if len(whichOrder) == 0:
            if targets is None: whichOrder = ['p', 'v']
            else: whichOrder = ['p', 'i', 'v']
        if not(targets is None):
            sampPoints = np.zeros((sampleSize, max(emRangeLength)), dtype = 'float')
            mostRanges = (ems[np.argmax(emRangeLength)]).ranges
            for i in range(sampPoints.shape[1]):
                sampPoints[:,i] = np.random.uniform(list(mostRanges.values())[i][0], list(mostRanges.values())[i][1], sampleSize)
            sampPoints = pd.DataFrame(sampPoints, columns = list(mostRanges.keys()))
            emImps = nthImplausible(ems, sampPoints, targets, cutoff = cutoff, getRaw = True)
            impOrder = np.apply_along_axis(sum, 0, emImps)
        emOrder = pd.DataFrame(data = {'p': [-1*erl for erl in emRangeLength], 'v': emRangeProds})
        if not(targets is None):
            emOrder['i'] = impOrder
        emOrder = emOrder.loc[:,whichOrder].sort_values(by = whichOrder, axis = 0)
        emOrderIndices = list(emOrder.index)
        return [ems[i] for i in emOrderIndices]
    return collectEmulators([em for emlist in ems for em in emlist], targets = targets,
                            cutoff = cutoff, ordering = ordering, sampleSize = sampleSize)

'''
    Get Parameter Ranges

    Determine mimimal or maximal parameter ranges for a collection of emulators.

    This is a more complex version of a call to em.ranges, which accommodates a variety
    of inputs to the ems argument (including nested and hierarchical emulator collections).
    Either the largest (most conservative) set of ranges or the smallest (most restrictive)
    are returned, depending on the value of minimal.

    This is used in point generation in two ways: for Latin Hypercube proposals, the minimal
    hyperrectangle is used; for importance sampling, the largest set of ranges is taken.

    Parameters:
    -----------
    ems: [Emulator] | [[Emulator]]
        The emulators in question
    minimal: bool
        Whether to return the smallest (True) or largest (False) hyperrectangle.
    
    Returns:
    --------
    A named list of ranges, each consisting of an upper and lower bound.
'''
def getRanges(ems, minimal = True):
    ems = collectEmulators(ems)
    rangeLengths = [len(list(em.ranges.values())) for em in ems]
    if not len(np.unique(rangeLengths)) == 1:
        ems = [em for em in ems if rangeLengths == max(rangeLengths)]
    rangeWidths = np.full((len(ems), max(rangeLengths)), 0, dtype = 'float')
    for i in range(len(ems)):
        for j in range(max(rangeLengths)):
            rangeWidths[i, j] = np.diff(list(ems[i].ranges.values())[j])
    if minimal:
        whichChoose = np.apply_along_axis(np.argmin, 0, rangeWidths)
    else:
        whichChoose = np.apply_along_axis(np.argmax, 0, rangeWidths)
    newRanges = [list(ems[whichChoose[i]].ranges.values())[i] for i in range(len(list(ems[0].ranges.values())))]
    return dict(zip(list(ems[0].ranges.keys()), newRanges))