"""Helper structs to store/maintain solutions and matrices."""

import math
import numpy as np
from scipy.special import comb


class Solution:
    '''
    B: dictionary. key = i
    T: dictionary. key = (i,j)
    '''

    def __init__(self, B, T, intercept=None, gap=None):
        # stores only non-zero indices/values.
        self.B = B.copy()
        self.T = T.copy()
        self.intercept = intercept
        self.gap = gap

    def ToArray(self, BIndices, TIndices):
        BArray = np.zeros(len(BIndices))
        for counter, index in enumerate(BIndices):
            if index in self.B:
                BArray[counter] = self.B[index]

        TArray = np.zeros(len(TIndices))
        for counter, index in enumerate(TIndices):
            if index in self.T:
                TArray[counter] = self.T[index]

        return (BArray, TArray)


class Matrices:
    def __init__(self, X, y, Bindices=[], Tindices=[], maxNorm=0):
        self.X = X
        self.y = y
        self.ybar = np.mean(self.y)
        self.ycentered = y - self.ybar
        # might need to change this in case materialization is used
        self.pMain = X.shape[1]
        self.P = comb(self.pMain, 2, exact=True)
        self.indexMapB = {Bindices[i]: i for i in range(
            len(Bindices))}  # maps var number to index
        self.indexMapT = {}  # maps var number to index
        self.XB = X[:, Bindices]
        self.XT = np.zeros(shape=(self.X.shape[0], len(Tindices)))
        l = 0
        for i, j in Tindices:
            self.XT[:, l] = X[:, i] * X[:, j]
            self.indexMapT[(i, j)] = l
            l += 1

        self.maxNorm = maxNorm

        self.bins = np.zeros(self.pMain, dtype=np.int64)
        for l in range(1, self.pMain):
            self.bins[l] = self.bins[l - 1] + self.pMain - l

        def interleave(p):
            for i in range(math.ceil(p / 2)):
                yield i
                yield p - i - 1

        self.balancedInd = list(interleave(self.pMain - 1))
        if len(self.balancedInd) != self.pMain - 1:
            self.balancedInd.pop()
        self.balancedInd = np.array(self.balancedInd)

    def Augment(self, Bindices, Tindices):
        BindicesNew = [i for i in Bindices if i not in self.indexMapB]
        TindicesNew = [key for key in Tindices if key not in self.indexMapT]

        pold = self.XB.shape[1]
        for i in range(len(BindicesNew)):
            self.indexMapB[BindicesNew[i]] = i + pold
        self.XB = np.hstack((self.XB, self.X[:, Bindices]))

        pold = self.XT.shape[1]
        XTtemp = np.zeros(shape=(self.X.shape[0], len(TindicesNew)))
        l = 0
        for i, j in TindicesNew:
            XTtemp[:, l] = self.X[:, i] * self.X[:, j]
            self.indexMapT[(i, j)] = l + pold
            l += 1
        self.XT = np.hstack((self.XT, XTtemp))

    def Retrieve(self, Bindices, Tindices):
        '''
        returns submatrices XB and XT corresponding to Bindices and Tindices
        '''
        Bidx = [self.indexMapB[i] for i in Bindices]
        Tidx = [self.indexMapT[i] for i in Tindices]
        return self.XB[:, Bidx], self.XT[:, Tidx]


class RelaxationVars:
    def __init__(
        self,
        BActive,
        TActive,
        initialSol,
        useDual,
        u,
        w,
        rold=np.zeros(1),
        maxrtX=None,
        Tviolating=set(),
            maxNorm=None):
        self.BActive = BActive  # the current active set -- type = list
        self.TActive = TActive  # the current active set -- type = list
        # initial sol to use for the convex solver -- type = Solution
        self.initialSol = initialSol
        self.u = u
        self.w = w
        self.useDual = useDual  # boolean to indicate whether to use a dual initialization
        self.rold = rold
        self.maxrtX = maxrtX
        self.Tviolating = Tviolating
        self.maxNorm = maxNorm


def FlatToTuple(x, data):
    """
    Converts a numpy array of flat indices to a python list of (i,j) pairs

    Derivation: (i+1)*(p-1) - i*(i-1)/2 + (j-i) - p = index => j = index - (i+1)*(p-1) + i*(i-1)/2 + i + p

    Test:
    for i in range(P):
        if FlatToTuple(np.array([i]))[0] != D[i + pMain]:
            Params["print"]("Problem!")
            Params["print"](i, FlatToTuple(np.array([i]))[0], D[i + pMain])
            break
    """
    iIndices = np.digitize(x, data.bins) - 1
    jIndices = x - (data.pMain - 1) * (iIndices + 1) + \
        iIndices * (iIndices - 1) / 2 + iIndices + data.pMain
    return list(zip(iIndices, jIndices.astype(int)))

def InitializeParams(params):
    default_params = {
        "M": 10**10,
        "alpha": 2,
        "tol": 1e-6,
        "tolCD": 1e-6,
        "CorrScreen": True,
        "nLambda": 100,
        "lambdaMinRatio": 0.05,
        "maxSuppSize": 500,
        "debug": False,
    }
    for key in default_params:
        if key not in params:
            params[key] = default_params[key]

    return params
