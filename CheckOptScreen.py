""" Checks the KKT conditions outside of the active set. """

from heapq import *
import os
import time
import warnings
import numpy as np
import numba
from numba import njit, prange
import networkx as nx
from gurobipy import *
from .Structs import *

from numba.errors import NumbaPerformanceWarning
os.environ["NUMBA_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@njit(numba.float64[::1](numba.float64[:,
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
    bins = binsin[:]  # used for better cache locality.
    rlTimesX = X.T * r
    rXijs = np.zeros(P)
    for l in prange(pMain - 1):
        i = balancedInd[l]
        # bins are used for load balancing.
        rXijs[bins[i]:bins[i + 1]] = np.dot(rlTimesX[i, :], X[:, (i + 1):])
    return rXijs

def CheckViolating(
        X,
        r,
        fixedSet,
        fixedZeroSet,
        Binnz,
        minheap,
        Tindices,
        Tviolating,
        Lambda1ovM):
    '''
    fixedSet.remove((-1,-1)) # hack for numba's empty list/set problem -- remove when numba fixes this
    fixedZeroSet.remove(-1)
    Binnz.remove(-1)
    minheap.pop()
    Tindices.pop()
    Tviolating.remove((-1,-1))
    '''
    feasible = True
    for l, j in Tviolating:
        rXi = np.abs(np.dot(r, X[:, l] * X[:, j]))
        Blnnz = l in Binnz
        # is either of l or j fixed to zero?
        fixedlj0 = (l in fixedZeroSet) or (j in fixedZeroSet)
        if ((l, j) not in fixedSet) and not fixedlj0:
            # can be improved later by splitting into several cases.
            if (not Blnnz) or (j not in Binnz):
                if rXi > Lambda1ovM:
                    minheap.append((rXi, (l, j)))
            # l and j are both nnz (and not fixed to zero).
            elif rXi > Lambda1ovM and (l, j) not in Tindices:
                # OK since l and j are already in activeset.
                Tindices.append((l, j))
                feasible = False
    return feasible


'''
def ComputeDots(X, r, pMain, P, binsin):
    bins = binsin[:] # used for better cache locality
    rlTimesX = X.T * r
    rXijs = np.zeros(P)
    for l in prange(pMain-1):
        rXijs[bins[l]:bins[l+1]] = np.dot(rlTimesX[l,:], X[:,(l+1):])
    return rXijs
'''

'''
@njit(parallel=True, fastmath=True, cache=True)
def ComputeDots(X, r, pMain, P):
    rlTimesX = X.T * r
    rXijs = np.zeros(P)
    for l in prange(pMain-1):
        start = int(l*(pMain - (l+1)/2))
        rXijs[start:(start+pMain-l-1)] = np.dot(rlTimesX[l,:], X[:,(l+1):])
    return rXijs
'''


def CheckOptScreen(
        data,
        Bindices,
        Tindices,
        currentB,
        currentT,
        r,
        rold,
        maxrtX,
        Tviolating,
        maxNorm,
        Params,
        BscreenIndices,
        fixedVars):
    '''
    Finds Bi's and Tij's (outside the Bindices and Tindices) that violate the optimality
    conditions. Main logic:
        What not to check?
        - If i in fixedVars, do not check Bi
        - If (i in fixedVars and zero) or (j in fixedVars and zero) then do not check Tij (since any nnz Tij is infeasible)

        What to check?
        - If i not in fixedVars and not in Bindices, check Bi
        - If Bi or Bj is zero, check Tij
        - If Bi and Bj are nnz, but (i,j) not in Tindices, check Tij

    Only a portion of the violations in each subgraph is appended to Bindices/Tindices (current threshold = 5) for each subgraph
    The appended Bindices/Tindices are sorted by decreasing order of |<r,Xi>|.

    Bindices and Tindices are sets of indices that are (potentially) modified.

    ToDo: Fix BscreenIndices (not an issue for now since we're using BscreenIndices = range(pMain))
    '''
    X = data.X
    pMain = data.pMain
    #D = data.D
    feasible = True  # stays true unless an infeasbility is found
    slackimp = True
    # set k to a relatively large number.
    k = 1000000
    M = Params["M"]
    Lambda0 = Params["Lambda"]
    Lambda1 = Params["alpha"] * Lambda0
    MovLambda = M / Lambda0
    LambdaovM = Lambda0 / M
    Lambda1ovM = Lambda1 / M

    Bind = {}
    for i in range(len(Bindices)):
        Bind[Bindices[i]] = i

    #p = pMain + len(D)
    rtXi = {}
    minheapmain = []

    # H. asm. below.
    # BCheckSet contains the indices of all the Bi's not in fixedVars (i.e.,
    # typically, [p] - fixedVars).
    BCheckSet = set(BscreenIndices).difference(set(fixedVars.keys()))
    for i in BCheckSet:
        Bizero = i not in Bindices or (
            i in Bindices and currentB[Bind[i]] == 0)
        if Bizero:
            rXi = np.abs(np.dot(r, X[:, i]))  # needed here for now.
            rtXi[i] = rXi
            if len(minheapmain) <= k:
                heappush(minheapmain, (rXi, i))
            else:
                if minheapmain[0][0] < rXi:
                    heappop(minheapmain)
                    heappush(minheapmain, (rXi, i))

    multiplierTimesLambda1ovM = 0.7 * Lambda1ovM
    firstRun = maxrtX is None
    if not firstRun:
        gamma = np.dot(rold, r) / np.dot(rold, rold)  # 1
        tempdiff = r - gamma * rold
        eps = np.sqrt(np.dot(tempdiff, tempdiff))
        #norm = max(L2Norms.values())
        Params["print"]("maxrtX: ", maxrtX)
        Params["print"]("(LambdaovM - eps*norm)/gamma: ",
                        (LambdaovM - eps * data.maxNorm) / np.abs(gamma))

    minheap = []

    '''
    Binnz = {} # indicates whether Bi is (nnz or in fixedVars) = True, or not (False)
    for i in BscreenIndices:
        Binnz[i] = (i in Bindices and currentB[Bind[i]] != 0) or (i in fixedVars)
    if not firstRun and maxrtX <= (LambdaovM - eps*data.maxNorm)/np.abs(gamma):
        Params["print"]("################## !!!!!!!!!!!!!!!! #####################")
        Params["print"]("################## !!!!!!!!!!!!!!!! #####################")
        Params["print"]("################## !!!!!!!!!!!!!!!! #####################")
        Params["print"]("################## !!!!!!!!!!!!!!!! #####################")
        Params["print"]("Avoided full evaluation!")

        for l,j in Tviolating:
            rXi = np.abs(np.dot(r, X[:,l]*X[:,j]))
            Blnnz = Binnz[l]
            # is either of l or j fixed to zero?
            fixedlj0 = (l in fixedVars and fixedVars[l] == 0) or (j in fixedVars and fixedVars[j] == 0)
            if ((l,j) not in fixedVars) and not fixedlj0:
                if (not Blnnz) or (not Binnz[j]): # can be improved later by splitting into several cases
                    if rXi > LambdaovM:
                        minheap.append((rXi,(l,j)))
                else: # l and j are both nnz (and not fixed to zero)
                    if rXi > LambdaovM and (l,j) not in Tindices: # add appx kkt here later
                        Tindices.append((l,j)) # OK since l and j are already in activeset.
                        Params["print"]("!!!!!!")
                        Params["print"]("!!!!!!")
                        Params["print"]("!!!!!! Appended T", (l,j))
                        Params["print"]("rXi: ", rXi, " LambdaovM: ", LambdaovM)
                        feasible = False
    '''

    Binnz = {i for i in BscreenIndices if (
        i in Bindices and currentB[Bind[i]] != 0) or (i in fixedVars)}

    if not firstRun and maxrtX <= (
            Lambda1ovM - eps * data.maxNorm) / np.abs(gamma):
        Params["print"](
            "################### !!!!!!!!!!!!!!!! #####################")
        Params["print"]("Avoided full evaluation!")

        start = time.time()
        fixedZeroSet = {
            key for key,
            val in fixedVars.items() if not isinstance(
                key,
                tuple) and val == 0}
        fixedSet = {key for key in fixedVars if isinstance(key, tuple)}
        end = time.time()
        Params["print"]("!!! Time for precheck: ", end - start)
        '''
        minheap.append((-1.1,(-1,-1)))
        Tindices.append((-1,-1))
        fixedSet.add((-1,-1))
        fixedZeroSet.add(-1)
        Binnz.add(-1)
        Tviolating.add((-1,-1))
        '''
        start = time.time()
        feasible = CheckViolating(
            X,
            r,
            fixedSet,
            fixedZeroSet,
            Binnz,
            minheap,
            Tindices,
            Tviolating,
            Lambda1ovM)
        end = time.time()
        Params["print"]("!!! Time for avoidance check: ", end - start)

    else:
        # '''
        rold = r

        rXijs = ComputeDots(
            X,
            r,
            data.pMain,
            data.P,
            data.bins,
            data.balancedInd)
        rXijs = np.abs(rXijs)

        ViolatingIndices = np.nonzero(rXijs > multiplierTimesLambda1ovM)[0]
        OptCheckIndices = np.nonzero(rXijs > Lambda1ovM)[0]

        #sortedIndices = np.argsort(rXijs)
        #np.searchsorted(rXijs, [], side="right", sorter=sortedIndices)

        # Remove the indices of (i,j)'s that are fixed or one of i and j is
        # fixed to zero.
        skipIndices = []
        for key in fixedVars:
            if not isinstance(key, tuple) and fixedVars[key] == 0:
                skipIndices = skipIndices + \
                    [(key, j) for j in range(key + 1, pMain)] + [(i, key) for i in range(0, key)]
            elif isinstance(key, tuple) and fixedVars[key] == 0:
                skipIndices.append(key)
        for counter, (i, j) in enumerate(skipIndices):
            skipIndices[counter] = int(
                (i + 1) * (pMain - 1) - i * (i - 1) / 2 + (j - i) - pMain)

        if len(skipIndices) != 0:
            ViolatingIndices = np.setdiff1d(ViolatingIndices, skipIndices)
            OptCheckIndices = np.setdiff1d(OptCheckIndices, skipIndices)

        Tviolating = set(FlatToTuple(ViolatingIndices, data))
        minheapindices = FlatToTuple(OptCheckIndices, data)

        minheapTemp = list(zip(rXijs[OptCheckIndices], minheapindices))
        for val, (i, j) in minheapTemp:
            if i in Binnz and j in Binnz:
                if (i, j) not in Tindices:
                    # OK since l and j are already in activeset.
                    Tindices.append((i, j))
                    Params["print"]("!!!!!!")
                    Params["print"]("!!!!!!")
                    Params["print"]("!!!!!! Appended T", (i, j))
                    Params["print"]("rXi: ", val, " Lambda1ovM: ", Lambda1ovM)
                    feasible = False
            else:
                minheap.append((val, (i, j)))

        mask = np.full(data.P, False)
        mask[ViolatingIndices] = True
        mask[skipIndices] = True
        rXijsMasked = np.ma.array(rXijs, mask=mask)  # copies by reference
        maxrtX = np.ma.max(rXijsMasked)
        # '''

    # Find the connected components

    start = time.time()
    G = nx.Graph()
    G.add_edges_from(v[1] for v in minheap)
    disconnectedGraphs = (G.subgraph(c).edges()
                          for c in nx.connected_components(G))
    end = time.time()
    Params["print"]("!!! Time for constructing graphs: ", end - start)

    # Check feasibility of the connected LPs.
    rtXij = {v[1]: v[0] for v in minheap}
    infeasibleGraphs = []
    for graph in disconnectedGraphs:
        Params["print"]("Graph Size: ", len(graph))
        lpModel = Model()
        lpModel.setParam('OutputFlag', False)
        mu = {}
        psi = {}
        for i, j in graph:
            izero = i not in Bindices or (
                i in Bindices and currentB[Bind[i]] == 0)
            izero = izero and (i not in fixedVars)

            jzero = j not in Bindices or (
                j in Bindices and currentB[Bind[j]] == 0)
            jzero = jzero and (j not in fixedVars)

            if izero:
                if i not in mu:
                    mu[i] = {}
                mu[i][j] = lpModel.addVar(vtype=GRB.CONTINUOUS, lb=0)

            if jzero:
                if j not in mu:
                    mu[j] = {}
                mu[j][i] = lpModel.addVar(vtype=GRB.CONTINUOUS, lb=0)

            if slackimp:
                psi[(i, j)] = lpModel.addVar(vtype=GRB.CONTINUOUS, lb=0)

            if izero and not jzero:
                lpModel.addConstr(mu[i][j] == (
                    M / Lambda0) * rtXij[(min(i, j), max(i, j))] - Lambda1 / Lambda0 - psi[(i, j)])

            elif jzero and not izero:
                lpModel.addConstr(mu[j][i] == (
                    M / Lambda0) * rtXij[(min(i, j), max(i, j))] - Lambda1 / Lambda0 - psi[(i, j)])

            elif izero and jzero:
                lpModel.addConstr(mu[i][j] +
                                  mu[j][i] == (M /
                                               Lambda0) *
                                  rtXij[(min(i, j), max(i, j))] -
                                  Lambda1 /
                                  Lambda0 -
                                  psi[(i, j)])

        for i in mu:
            if not slackimp:
                lpModel.addConstr(
                    quicksum(mu[i].values()) <= 1 - (M / Lambda0) * rtXi[i])
            else:
                if (M / Lambda0) * rtXi[i] <= 1:
                    lpModel.addConstr(
                        quicksum(mu[i].values()) - 1 + (M / Lambda0) * rtXi[i] <= 0)

        if not slackimp:
            lpModel.setObjective(0, GRB.MINIMIZE)
            lpModel.update()
            lpModel.optimize()
            if lpModel.Status == 3:
                infeasibleGraphs.append(graph)
                lpModel.feasRelaxS(0, False, False, True)
                Params["print"]("Optimizing for infeasibility")
                lpModel.optimize()
                obj = lpModel.getObjective()
                if obj.getValue() > 0.05:
                    feasible = False
                else:
                    Params["print"]("Opt. conditions appx. satisfied")
                # break
        else:
            eta = lpModel.addVar(vtype=GRB.CONTINUOUS, lb=0)
            for (i, j) in psi:
                lpModel.addConstr(eta >= psi[(i, j)])
            lpModel.setObjective(eta, GRB.MINIMIZE)
            lpModel.update()
            lpModel.optimize()
            obj = lpModel.getObjective()
            if obj.getValue() > 0.05:
                feasible = False
                infeasibleGraphs.append(graph)
            else:
                Params["print"]("Opt. conditions satisfied")

    Params["print"]("Feasible Graphs?: ", feasible)

    # Keep the following after checking graph feasibility (since graph feas.
    # depends on current Bindices).
    threshold = 10
    minheap = []
    for i in rtXi:
        if len(minheap) < threshold:
            heappush(minheap, (rtXi[i], i))
        else:
            if minheap[0][0] < rtXi[i]:
                heappop(minheap)
                heappush(minheap, (rtXi[i], i))
    for v in sorted(minheap, reverse=True):
        i = v[1]
        if (M / Lambda0) * rtXi[i] >= 1:
            if i not in Bindices:
                feasible = False
                Bindices.append(i)
                Params["print"]("!!!!!!")
                Params["print"]("!!!!!!")
                Params["print"]("!!!!!! Appended B", i)
                Params["print"]("rXi: ", rtXi[i], " LambdaovM: ", LambdaovM)

    Params["print"]("Feasible? ", feasible)
    if not feasible:
        for graph in infeasibleGraphs:
            minheap = []
            for pair in graph:
                i = min(pair[0], pair[1])
                j = max(pair[0], pair[1])
                mag = rtXij[(i, j)]
                if (i, j) not in Tindices:
                    if len(minheap) < threshold:
                        heappush(minheap, (mag, (i, j)))
                    else:
                        if minheap[0][0] < mag:
                            heappop(minheap)
                            heappush(minheap, (mag, (i, j)))

            for v in sorted(minheap, reverse=True):
                pair = v[1]
                i = min(pair[0], pair[1])
                j = max(pair[0], pair[1])
                # index not in Tindices by construction of minheap.
                Tindices.append((i, j))
                if i not in Bindices:
                    Bindices.append(i)
                if j not in Bindices:
                    Bindices.append(j)

    return (feasible, Bindices, Tindices, rold, maxrtX, Tviolating, maxNorm)
