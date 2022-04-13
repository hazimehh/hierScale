"""Computes the initial active set and lambda_max

This runs before solving for the first solution in the path.

"""

from heapq import *
import os
import numpy as np
import numba
from numba import njit, prange
from .Structs import *
os.environ["NUMBA_WARNINGS"] = "1"

@njit(numba.types.Tuple((numba.float64[::1],
                         numba.float64))(numba.float64[:,
                                                       ::1],
                                         numba.float64[::1],
                                         numba.int64,
                                         numba.int64,
                                         numba.int64[::1],
                                         numba.int64[::1]),
      parallel=True,
      fastmath=True,
      cache=True)
def ComputeDots(X, r, pMain, P, binsin, balancedInd):
    bins = binsin[:]
    rXijs = np.zeros(P)
    l2Norms = np.zeros(P)
    XT = X.T.copy()
    for l in prange(pMain - 1):
        i = balancedInd[l]
        # (X[:,(l+1):].T * X[:,l]).T
        XlXjs = np.transpose(XT[(i + 1):, :] * X[:, i])
        # XlXjs[:,:] is needed because of numba  #np.linalg.norm(XlXjs, axis=0)
        l2Norms[bins[i]:bins[i + 1]] = np.sum(XlXjs[:, :] ** 2, axis=0)
        rXijs[bins[i]:bins[i + 1]] = np.dot(r.T, XlXjs)

    maxNorm = np.sqrt(np.max(l2Norms))
    rXijs = np.abs(rXijs)
    return (rXijs, maxNorm)


'''
def ComputeDots(X, r, pMain, P, binsin):
    bins = binsin[:]
    rXijs = np.zeros(P)
    l2Norms = np.zeros(P)
    XT = X.T.copy()
    for l in prange(pMain-1):
        #start = #int(l*(pMain - (l+1)/2))
        #end = #start+pMain-l-1
        XlXjs = np.transpose(XT[(l+1):,:] * X[:,l]) #(X[:,(l+1):].T * X[:,l]).T
        l2Norms[bins[l]:bins[l+1]] = np.sum(XlXjs[:,:] ** 2, axis=0) #XlXjs[:,:] is needed because of numba  #np.linalg.norm(XlXjs, axis=0)
        rXijs[bins[l]:bins[l+1]] = np.dot(r.T, XlXjs)

    maxNorm = np.sqrt(np.max(l2Norms))
    rXijs = np.abs(rXijs)
    return (rXijs, maxNorm)



'''


def InitialRun(data, kmain, kint, Params):
    # Later: Make this return currentB, currentT
    X = data.X
    y = data.y
    pMain = data.pMain
    r = y - data.ybar
    Bindices = []
    Tindices = []
    rtXi = {}
    minheapmain = []
    # In what follows, we assume beta_i = 0 => theta_{i:} = 0.
    # Can handle this later -- no need for it now.
    for i in range(pMain):
        rXi = np.abs(np.dot(r, X[:, i]))  # needed here for now..
        rtXi[i] = rXi
        if len(minheapmain) < kmain:
            heappush(minheapmain, (rXi, i))
        else:
            if minheapmain[0][0] < rXi:
                heappop(minheapmain)
                heappush(minheapmain, (rXi, i))

    minheap = []
    '''
    for i in range(pMain,p):
        l,j = D[i] # (l,j) are the main effects corresponding to i
        rXi = np.abs(np.dot(r, X[:,l]*X[:,j]))
        if len(minheap) < k:
            heappush(minheap,(rXi,D[i]))
        else:
            if minheap[0][0] < rXi:
                heappop(minheap)
                heappush(minheap,(rXi,D[i]))
    '''
    '''
    keys = {(i,j) for i in range(pMain-1) for j in range(i+1,pMain)}
    L2Norms = dict.fromkeys(keys) # allocate space for the dict initially
    for l in range(pMain-1):
        #rlTimesXl = r*X[:,l]
        #rXis = np.abs(np.dot(rlTimesXl.T, X[:,(l+1):]))
        XlXjs = (X[:,(l+1):].T * X[:,l]).T
        rXis = np.abs(np.dot(r.T, XlXjs))
        L2NormsTemp = np.linalg.norm(XlXjs, axis=0) # finds the l2 norm of every column
        for j in range(l+1, pMain):
            #rXi = rXis[j-l-1] #np.abs(np.dot(X[:,j], rlTimesXl))
            index = j-l-1
            XlXj = XlXjs[:,index]
            rXi = rXis[index] #np.abs(np.dot(r.T,XlXj))
            if len(minheap) < kint:
                heappush(minheap,(rXi,(l,j)))
            else:
                if minheap[0][0] < rXi:
                    heappop(minheap)
                    heappush(minheap,(rXi,(l,j)))
            L2Norms[(l,j)] = L2NormsTemp[index] #np.sqrt(np.dot(XlXj,XlXj))
    maxNorm = max(L2Norms.values())
    data.maxNorm = maxNorm
    '''
    rXijs, maxNorm = ComputeDots(
        X, r, pMain, data.P, data.bins, data.balancedInd)
    data.maxNorm = maxNorm
    topIndicesFlat = np.argpartition(rXijs, -kint)[-kint:]
    topValues = rXijs[topIndicesFlat]
    topIndices = FlatToTuple(topIndicesFlat, data)
    minheap = list(zip(topValues, topIndices))

    for pair in sorted(minheapmain, reverse=True):
        i = pair[1]
        if i not in Bindices:
            Bindices.append(i)

    for pair in sorted(minheap, reverse=True):
        i = pair[1][0]
        j = pair[1][1]
        if i not in Bindices:
            Bindices.append(i)
        if j not in Bindices:
            Bindices.append(j)
        #index = int((i+1)*(pMain-1) - i*(i-1)/2 + (j-i))
        # if index not in Tindices:
        #    Tindices.append(index)
        if (i, j) not in Tindices:
            Tindices.append((i, j))

    #Bindices = sorted(Bindices)
    #Tindices = sorted(Tindices)

    # lambdaMax = max(max(minheapmain)[0], max(minheap)[0])*Params["M"] #
    # *Params["M"] since lambdaMax is lambda0Max

    # *Params["M"] since lambdaMax is lambda0Max
    lambdaMax = max(minheapmain)[0] * Params["M"]

    return (Bindices, Tindices, lambdaMax, maxNorm)
