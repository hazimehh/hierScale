"""Implementation of Proximal Gradient Descent with Dual Block Coordinate Ascent.
Proximal screening rules are employed to identify zero interaction coordinates
and blocks.
"""

import time
import math
from copy import copy
from collections import defaultdict
import numpy as np
from gurobipy import *
import numba
from numba import njit
from .Structs import *


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def T(
        wij,
        wji,
        thetaij,
        M,
        Lambda0,
        L,
        frac,
        frac1,
        Mpfrac1,
        LambdaovM,
        Lambda1ovM):
    h = thetaij - frac * (wij + wji)
    habs = np.abs(h)
    if habs <= frac1:
        out = 0
    elif habs <= Mpfrac1:
        out = (habs - frac1) * np.sign(h)
    elif h >= Mpfrac1:
        out = M
    elif h <= -Mpfrac1:
        out = -M
    return (L / 2) * (out - thetaij)**2 + LambdaovM * \
        out * (wij + wji) + Lambda1ovM * np.abs(out)


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def q(ui, betai, M, Lambda0, L, frac):
    #frac = Lambda0/(M*L)
    if ui >= (betai - M) / frac and ui <= (betai + M) / frac:
        out = - frac * frac * L * ui * ui / 2 + (Lambda0 / M) * betai * ui
    elif ui <= (betai - M) / frac:
        out = (L / 2) * (M - betai)**2 + Lambda0 * ui
    elif ui >= (betai + M) / frac:
        out = (L / 2) * (M + betai)**2 - Lambda0 * ui
    return out


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def delT(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1, LambdaovM):
    h = thetaij - frac * (wij + wji)
    habs = np.abs(h)
    if habs <= frac1:
        out = 0
    elif habs <= Mpfrac1:
        out = (habs - frac1) * np.sign(h) * LambdaovM
    elif h >= Mpfrac1:
        out = Lambda0
    elif h <= -Mpfrac1:
        out = -Lambda0
    return out


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def delq(ui, betai, M, Lambda0, L, frac, Mpfrac, fracsqL, LambdaovM):
    #frac = Lambda0/(M*L)
    if ui >= (betai - M) / frac and ui <= (betai + M) / frac:
        out = - fracsqL * ui + LambdaovM * betai
    elif ui <= (betai - M) / frac:
        out = Lambda0
    elif ui >= (betai + M) / frac:
        out = -Lambda0
    return out


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def dualtoprimalu(ui, betai, M, Lambda0, L, frac):
    #frac = Lambda0/(M*L)
    if ui >= (betai - M) / frac and ui <= (betai + M) / frac:
        out = betai - frac * ui
    elif ui <= (betai - M) / frac:
        out = M
    elif ui >= (betai + M) / frac:
        out = -M
    return out


@njit(
    numba.float64(
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64),
    fastmath=True,
    cache=True)
def dualtoprimalw(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1):
    #frac = Lambda0/(M*L)
    h = thetaij - frac * (wij + wji)
    habs = np.abs(h)
    if habs <= frac1:
        out = 0
    elif habs <= Mpfrac1:
        out = (habs - frac1) * np.sign(h)
    elif h >= Mpfrac1:
        out = M
    elif h <= -Mpfrac1:
        out = -M
    return out


def terminate(oldObj, newObj, tol):
    # handle the case of all zeros optimal solution.
    if oldObj - newObj == 0:
        return True
    elif oldObj == -math.inf or oldObj == math.inf:
        return False
    elif np.abs(oldObj - newObj) / oldObj < tol:
        return True
    else:
        return False


@njit(numba.float64[::1](numba.float64[::1]), fastmath=True, cache=True)
def project(v):
    # projects on the l1-ball
    vmod = np.abs(v)
    nz = vmod[vmod.nonzero()]
    s = np.sort(nz)[::-1]
    cs = s.cumsum()
    if cs[-1] <= 1:
        return v
    ks = 1 + np.arange(len(s))
    idx = np.searchsorted(s * ks <= cs - 1, True) - 1
    lmbda = (cs[idx] - 1) / (1 + idx)
    thresh_mult = np.maximum(1 - lmbda / vmod, 0)
    return thresh_mult * v


def PGD(Params, relaxationVars, fixedBs, fixedTs, data):
    """
    Runs proximal gradient descent to solve the interactions problem on a specified active set.
    The inputs are not modified by PGD.
    """
    Tol = Params["tol"]
    TolCD = Params["tolCD"]
    Lambda0 = Params["Lambda"]
    Lambda1 = Params["alpha"] * Lambda0
    M = Params["M"]
    y = data.ycentered  # data.y - data.ybar

    Bindices = relaxationVars.BActive.copy()  # list
    Tindices = relaxationVars.TActive.copy()  # list of tuples (i,j)
    currentB, currentT = relaxationVars.initialSol.ToArray(Bindices, Tindices)
    fixedB = fixedBs.copy()  # Dict. key = index, value = 0 or 1 (no index if not fixed)
    fixedT = fixedTs.copy()  # Dict. key = (i,j), value = 0 or 1 (no index if not fixed)
    DualInitial = relaxationVars.useDual

    # Store the index mappings
    Bmap = {}  # Bmap[i] = index of i in currentB or XB
    for i in range(len(Bindices)):
        Bmap[Bindices[i]] = i

    Tmap = {}  # Tmap[(i,j)] = index of interaction in XT and currentT
    for i in range(len(Tindices)):
        c1, c2 = Tindices[i]
        Tmap[(c1, c2)] = i
        Tmap[(c2, c1)] = i

    # Next: Some sanity checks (those can be removed if we're carful about the
    # inputs)

    # Make sure if B_i is fixed to 0 then all T_{ij}'s (in Tindices) are also
    # fixed to zero
    for i, val in fixedB.items():
        if val == 0:
            for l, j in Tmap:
                if l < j and (l == i or j == i):
                    fixedT[(l, j)] = 0

    # Make sure if T_{ij} is fixed to 1 then both B_i and B_j are fixed to 1
    for key, val in fixedT.items():
        if val == 1:
            i, j = key
            fixedB[i] = 1
            fixedB[j] = 1

    # Delete from Bindices and Tindices all the indices s.t. z_i = 0 / z_{ij}
    # = 0
    Bzeros = []
    for i, val in fixedB.items():
        if val == 0:
            Bzeros.append(Bmap[i])
    for i in sorted(Bzeros, reverse=True):
        del Bindices[i]
    currentB = np.delete(currentB, Bzeros)

    Tzeros = []
    for key, val in fixedT.items():
        if val == 0:
            Tzeros.append(Tmap[key])
    for i in sorted(Tzeros, reverse=True):
        del Tindices[i]
    currentT = np.delete(currentT, Tzeros)

    # Update the index mappings
    Bmap = {}  # Bmap[i] = index of i in currentB or XB
    for i in range(len(Bindices)):
        Bmap[Bindices[i]] = i

    Tmap = {}  # Tmap[(i,j)] = index of interaction in XT and currentT
    for i in range(len(Tindices)):
        c1, c2 = Tindices[i]
        Tmap[(c1, c2)] = i
        Tmap[(c2, c1)] = i

    # End of sanity checks

    # Retrive the matrices of the optimization variables
    # Later: We can store the centered columns (but this will require twice
    # the memory)
    XB, XT = data.Retrieve(Bindices, Tindices)
    XBMean = XB.mean(axis=0)
    XB = XB - XBMean
    XTMean = XT.mean(axis=0)
    XT = XT - XTMean

    Bfree = [i for i in Bindices if i not in fixedB]
    Tfree = [(i, j) for i, j in Tmap if i < j and (i, j) not in fixedT]
    TfreeIndices = [Tmap[(i, j)]
                    for i, j in Tmap if i < j and (i, j) not in fixedT]
    lenFixedB = len(Bindices) - len(Bfree)
    lenFixedT = len([key for key in fixedT if fixedT[key] == 1])

    # (Dual) Block CD Variables
    u = defaultdict(float)
    w = defaultdict(dict)
    if not DualInitial:
        for i in Bindices:
            u[i] = 0
        for pair in Tmap:
            i, j = pair
            w[i][j] = 0
    else:
        for i in Bindices:
            if i in relaxationVars.u and i not in fixedB:
                u[i] = relaxationVars.u[i]
            else:
                u[i] = 0
        for i, j in Tmap:
            if j in relaxationVars.w[i] and (min(i, j), max(
                    i, j)) not in fixedT and i not in fixedB and j not in fixedB:
                w[i][j] = relaxationVars.w[i][j]
            else:
                # Important: we need w[i][j] = 0 if T_{ij} if fixed (this is
                # due to the thresholding function)
                w[i][j] = 0

    sortedIndices = {i: sorted(w[i]) for i in w}
    sortedIndices = defaultdict(list, sortedIndices)

    # Prepare all the fixed matrices/vectors required for grad evaluation
    # later.
    XBty = np.dot(XB.T, y)
    XBtXB = np.dot(XB.T, XB)
    XTty = np.dot(XT.T, y)
    XTtXT = np.dot(XT.T, XT)
    XBtXT = np.dot(XB.T, XT)

    # Compute the lipschitz constant of the grad.
    Xfull = np.hstack((XB, XT))
    if Xfull.shape[1] != 0:
        eigvals, v = np.linalg.eig(np.dot(Xfull.T, Xfull))
        L = np.max(np.real(eigvals))
    else:
        L = 1  # any value here should suffice - it's not used.

    # Compute the lipschitz constants for BCD.
    LCD = {}
    for i in Bindices:
        LCD[i] = (len(w[i]) + 1) * ((Lambda0**2) / (L * M**2))

    # Define the thresholding constants
    frac = Lambda0 / (M * L)
    Mpfrac = M + frac
    frac1 = Lambda1 / (M * L)
    Mpfrac1 = M + frac1
    fracsqL = frac * frac * L
    LambdaovM = Lambda0 / M
    Lambda1ovM = Lambda1 / M
    Lambda1ovLambda0 = Lambda1 / Lambda0

    start = time.time()

    oldObj = math.inf
    for it in range(5000):
        grad_B = - XBty + np.dot(XBtXB, currentB) + np.dot(XBtXT, currentT)
        grad_T = - XTty + np.dot(XTtXT, currentT) + np.dot(XBtXT.T, currentB)
        Bstar = currentB - grad_B / L
        Tstar = currentT - grad_T / L
        # Iterate over the blocks, running dual BCD.
        # We employ dual warm starts by using the same (u,w) across the PGD updates.
        CDPrevObj = -math.inf
        LCDCurrent = copy(LCD)
        useZeroSuffCondition = True
        if useZeroSuffCondition:
            # Perform proximal screening below.
            zeroGroups = set()
            for i in Bfree:
                zeroSufficient = False
                cumsum = 0
                for j in w[i]:
                    thrshld = max(
                        (abs(Tstar[Tmap[(i, j)]]) / frac - Lambda1ovLambda0), 0)
                    # Do feature level screening below.
                    if thrshld == 0:
                        # The initialization below ensures that \theta_{ij} is
                        # never updated by BCA.
                        w[i][j] = 0
                        w[j][i] = 0
                    else:
                        cumsum += thrshld

                if cumsum <= 1 - abs(Bstar[Bmap[i]]) / frac:
                    zeroSufficient = True
                if zeroSufficient:
                    u[i] = Bstar[Bmap[i]] / frac
                    for j in w[i]:
                        if abs(Tstar[Tmap[(i, j)]]) > frac1:
                            w[i][j] = Tstar[Tmap[(
                                i, j)]] / frac - Lambda1ovLambda0 * np.sign(Tstar[Tmap[(i, j)]])
                        else:
                            w[i][j] = 0
                        w[j][i] = 0
                        # Not nec. but can improve speed.
                        LCDCurrent[j] -= (Lambda0**2) / (L * M**2)
                    zeroGroups.add(i)

            BfreeMinusZeroGroups = [i for i in Bfree if i not in zeroGroups]
            CDObjConst = 0
            '''
            for i in zeroGroups:
                CDObjConst += q(u[i], Bstar[Bmap[i]], M, Lambda0, L,frac)
                for j in w[i]:
                    if i < j:
                        # T(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1, LambdaovM, Lambda1ovM)
                        CDObjConst += T(w[i][j], w[j][i], Tstar[Tmap[(i,j)]], M, Lambda0, L,frac, frac1, Mpfrac1, LambdaovM, Lambda1ovM)
            '''
            ####
        else:
            zeroGroups = set()
            CDObjConst = 0
            BfreeMinusZeroGroups = Bfree
        # To Turn the part above off, comment it out and set the following:
        # zeroGroups = set()
        # CDObjConst = 0
        # BfreeMinusZeroGroups = Bfree

        for innerit in range(10000):
            # for i in Bfree:
            for i in BfreeMinusZeroGroups:
                # First, Calculate utilde and wtilde for ith block
                utilde = u[i] + delq(u[i],
                                     Bstar[Bmap[i]],
                                     M,
                                     Lambda0,
                                     L,
                                     frac,
                                     Mpfrac,
                                     fracsqL,
                                     LambdaovM) / LCDCurrent[i]

                #wtilde = {}
                # for j in w[i]:
                # if B_j is fixed to 1, then we already set w[j][i] = 0
                #    wtilde[j] = w[i][j] + delT(w[i][j], w[j][i], Tstar[Tmap[(i,j)]], M, Lambda0, L,frac,  Mpfrac, fracsqL, LambdaovM)/LCD[i]
                sortedIndicesi = sortedIndices[i]
                # delT(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1, LambdaovM)
                wtilde = [w[i][j] + delT(w[i][j],
                                         w[j][i],
                                         Tstar[Tmap[(i,
                                                     j)]],
                                         M,
                                         Lambda0,
                                         L,
                                         frac,
                                         frac1,
                                         Mpfrac1,
                                         LambdaovM) / LCDCurrent[i] for j in sortedIndicesi]

                x = np.empty(shape=len(wtilde) + 1)
                # Solve the l1 projection problem.
                x[0] = utilde
                x[1:] = np.array(wtilde)
                projection = project(x)
                # Update the solution.
                u[i] = projection[0]
                # for j in range(len(w[i])):
                # w[i][sortedIndicesi[j]] = projection[j+1] ## +1 since u[i] is
                # first
                for counter, j in enumerate(sortedIndicesi):
                    w[i][j] = projection[counter + 1]
            # Calculate the current objective
            CDObj = CDObjConst  # 0
            for i in BfreeMinusZeroGroups:  # Bfree:
                CDObj += q(u[i], Bstar[Bmap[i]], M, Lambda0, L, frac)
                for j in w[i]:
                    if i < j:
                        # T(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1, LambdaovM, Lambda1ovM)
                        CDObj += T(w[i][j], w[j][i], Tstar[Tmap[(i, j)]], M,
                                   Lambda0, L, frac, frac1, Mpfrac1, LambdaovM, Lambda1ovM)
            #Params["print"]("Inner obj: ", CDObj)
            if terminate(CDPrevObj, CDObj, TolCD):
                break
            CDPrevObj = CDObj

        # Get back the primal solution.
        for i in range(len(Bindices)):
            # if Bindices[i] is fixed to 1, then u[Bindices[i]] = 0 and the
            # update below will lead to currentB[i] = Bstar[i] (or +- M)
            if Bindices[i] not in zeroGroups:
                # assuming Bindices is sorted
                currentB[i] = dualtoprimalu(
                    u[Bindices[i]], Bstar[i], M, Lambda0, L, frac)
            else:
                currentB[i] = 0

        for i, j in Tmap:
            # if i or j is fixed, the corresponding w[i][j] will be zero, which
            # leads to the correct update.
            if i < j:
                if (i, j) in Tfree:
                    # dualtoprimalw(wij, wji, thetaij, M, Lambda0, L, frac, frac1, Mpfrac1)
                    if i in zeroGroups or j in zeroGroups:
                        currentT[Tmap[(i, j)]] = 0
                    else:
                        currentT[Tmap[(i, j)]] = dualtoprimalw(
                            w[i][j], w[j][i], Tstar[Tmap[(i, j)]], M, Lambda0, L, frac, frac1, Mpfrac1)
                else:  # careful, this is the case when no thresholding should be applied
                    coefficient = Tstar[Tmap[(i, j)]]
                    if np.abs(coefficient) <= M:
                        currentT[Tmap[(i, j)]] = coefficient
                    else:
                        currentT[Tmap[(i, j)]] = M * np.sign(coefficient)

        r = y - np.dot(XB, currentB) - np.dot(XT, currentT)

        maxterm = 0
        for i in range(len(currentB)):
            if Bindices[i] not in fixedB:
                maxtemp = np.abs(currentB[i])
                for j in w[Bindices[i]]:
                    maxtemp = max(maxtemp, np.abs(
                        currentT[Tmap[(Bindices[i], j)]]))
                maxterm += maxtemp
        l1norm = np.sum(np.abs(currentT[TfreeIndices]))
        # IMPORTANT: Avoid using lenFixed and lenFixedT here.....!!!!!! ####
        currentobjective = 0.5 * np.dot(r, r) + Lambda0 * (
            lenFixedB + lenFixedT) + (Lambda0 / M) * maxterm + (Lambda1 / M) * l1norm

        if currentobjective > oldObj:
            Params["print"]("Objective Increased!!!")

        if terminate(oldObj, currentobjective, Tol):
            break

        oldObj = currentobjective
        Params["print"]("Iteration :", it, ". Objective: ", currentobjective)

    end = time.time()
    Params["print"]("Time: ", end - start, " seconds.")

    # Check if any small values should be zero.
    # Start with more aggressive checks first.
    Trunc = False
    for epsilon in [0.01, 1e-3, 1e-4, 1e-5, 1e-6]:
        currentBtrunc = np.copy(currentB)
        currentTtrunc = np.copy(currentT)
        currentBSetToZero = np.nonzero(np.abs(currentB) < epsilon)[0]
        currentBtrunc[currentBSetToZero] = 0
        currentBSetToZeroPSet = set(currentBSetToZero)
        for (i, j) in Tmap:
            if Bmap[i] in currentBSetToZeroPSet or Bmap[j] in currentBSetToZeroPSet:
                currentTtrunc[Tmap[(i, j)]] = 0

        currentTtrunc[np.abs(currentT) < epsilon] = 0
        rtrunc = y - np.dot(XB, currentBtrunc) - np.dot(XT, currentTtrunc)
        maxterm = 0
        for i in range(len(currentBtrunc)):
            if Bindices[i] not in fixedB:
                maxtemp = np.abs(currentBtrunc[i])
                for j in w[Bindices[i]]:
                    maxtemp = max(maxtemp, np.abs(
                        currentTtrunc[Tmap[(Bindices[i], j)]]))
                maxterm += maxtemp
        l1norm = np.sum(np.abs(currentTtrunc[TfreeIndices]))
        objectivetrunc = 0.5 * np.dot(rtrunc, rtrunc) + Lambda0 * (
            lenFixedB + lenFixedT) + (Lambda0 / M) * maxterm + (Lambda1 / M) * l1norm

        Params["print"](
            "eps: ",
            epsilon,
            " objectivetrunc:  ",
            objectivetrunc,
            "  currentobjective: ",
            currentobjective)
        # 1.01 might be beneficial in some extreme cases where supp becomes
        # very large (but might also cause descent problems)
        if objectivetrunc <= currentobjective:
            '''
            currentB = currentBtrunc
            currentT = currentTtrunc
            r = rtrunc
            currentobjective = objectivetrunc
            '''
            Params["print"]("###CHANGE###", "eps: ", epsilon)
            Params["print"]("Final Objective :", objectivetrunc)
            Trunc = True
            break

    integral = True

    for i in Bfree:
        zi = np.abs(currentB[Bmap[i]]) / M
        if zi > 0 and zi < 0.999:
            integral = False

    for i in TfreeIndices:
        zi = np.abs(currentT[i]) / M
        if zi > 0 and zi < 0.999:
            integral = False

    Bnnz = {key: currentB[Bmap[key]]
            for key in Bmap if currentB[Bmap[key]] != 0}
    Tnnz = {(i, j): currentT[Tmap[(i, j)]]
            for i, j in Tmap if i < j and currentT[Tmap[(i, j)]] != 0}
    intercept = data.ybar - np.dot(XBMean, currentB) - np.dot(XTMean, currentT)
    sol = Solution(Bnnz, Tnnz, intercept)

    if Trunc:
        BnnzTrunc = {key: currentBtrunc[Bmap[key]]
                     for key in Bmap if currentBtrunc[Bmap[key]] != 0}
        TnnzTrunc = {(i, j): currentTtrunc[Tmap[(
            i, j)]] for i, j in Tmap if i < j and currentTtrunc[Tmap[(i, j)]] != 0}
        interceptTrunc = data.ybar - \
            np.dot(XBMean, currentBtrunc) - np.dot(XTMean, currentTtrunc)
        solTrunc = Solution(BnnzTrunc, TnnzTrunc, interceptTrunc)
    else:
        BnnzTrunc = Bnnz
        TnnzTrunc = Tnnz
        interceptTrunc = intercept
        solTrunc = sol

    return (sol, solTrunc, currentobjective, integral, r, u, w)
