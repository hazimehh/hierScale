"""Runs the solver for a single lambda in the path."""
from .RelaxationSolver import PGD
from .CheckOptScreen import CheckOptScreen
from .Structs import *


def FitSingleLambda(Params, relaxationVars, fixedVars, data):
    """
    Fits the convex relaxation of the strong hierarchy model for a single value of the regularization parameter lambda.

    Parameters:
        Params (dict): A dictionary of parameters with the following keys
            fixedVars (dict)
            DualInitial (bool): indicates whether to use dual initialization on the first call to PGD.
            Bindices (list): list of the Bi's to consider in the active set.
            Tindices (list): list of the Tij's to consider in the active set. Each element is of the form (i,j).
            currentB (np.array): coefficient values corresponding to Bindices.
            currentT (np.array): coefficient values corresponding to Tindices.
            u (dict): dictionary of the dual variables corresponding to Bi's. key = index (i.e., i) and value = coefficient value.
            w (dict): dictionary of the dual variables corresponding to Tij's. key = (i,j) and value = coefficient value.
            rold (np.array): the vector of residuals at which the last **full** check of the optimality conditions was done.
            maxrtX (float): the maximum of r'X at the last full check of the opt. conditions.
            Tviolating (list): the indices of the violating Tij's in the last full check of optimality conditions.
            maxNorm: the max l2 norm of the non-violating columns in the last full check of optimality conditions.

    Output: A tuple (B, T, objective, integral)
            Also, Params will be modified such that Bindices, Tindices, currentB, currentT, u, v, rold, maxrtX, Tviolating, and maxNorm reflect
            the optimal solution obtained at the end of the fitting process.
    """

    # Split fixedVars into fixedB and fixedT
    fixedB = {
        key: val for key,
        val in fixedVars.items() if not isinstance(
            key,
            tuple)}  # == int can cause problems
    fixedT = {
        key: val for key,
        val in fixedVars.items() if isinstance(
            key,
            tuple)}
    #Params["print"]("fixedVars: ", fixedVars)
    #Params["print"]("fixedB: ", fixedB)
    #Params["print"]("fixedT: ", fixedT)
    Bindices = relaxationVars.BActive
    Tindices = relaxationVars.TActive

    feasible = False
    while not feasible:
        sol, solTrunc, obj, integral, r, u, w = PGD(
            Params, relaxationVars, fixedB, fixedT, data)
        ###
        # This block is for debugging only
        '''
        B, T, obj, integral, r = RunRelaxation(Params, Bindices.copy(), Tindices.copy(), fixedVars, data)
        sol = Solution(B,T)
        u = {} ####
        w = {} ####
        '''
        ###
        currentB, currentT = sol.ToArray(Bindices, Tindices)

        oldBindices = Bindices.copy()
        oldTindices = Tindices.copy()

        # New imp using CheckOptScreen instead of CheckOpt.
        feasible, Bindices, Tindices, rold, maxrtX, Tviolating, maxNorm = CheckOptScreen(data, Bindices.copy(), Tindices.copy(
        ), currentB, currentT, r, relaxationVars.rold, relaxationVars.maxrtX, relaxationVars.Tviolating, relaxationVars.maxNorm, Params, range(data.pMain), fixedVars)
        relaxationVars.rold = rold
        relaxationVars.maxrtX = maxrtX
        relaxationVars.Tviolating = Tviolating
        relaxationVars.maxNorm = maxNorm

        # update currentB and currentT (since Bindices and Tindices might get
        # updated).
        currentB, currentT = sol.ToArray(Bindices, Tindices)

        if not feasible:
            Params["print"](
                "Optimality conditions not satisfied outside active set!!!")

            additionalBindices = list(
                set(Bindices).difference(
                    set(oldBindices)))
            additionalTindices = list(
                set(Tindices).difference(
                    set(oldTindices)))

            if len(additionalBindices) + len(additionalTindices) == 0:
                Params["print"]("Obtained same supp. Skipping...")
                break

            # force using dual init after the first call.
            relaxationVars.useDual = True

            Params["print"]("Augmenting Data...")
            data.Augment(additionalBindices, additionalTindices)

        relaxationVars.BActive = Bindices
        relaxationVars.TActive = Tindices
        relaxationVars.initialSol = sol
        relaxationVars.u = u
        relaxationVars.w = w

    return (solTrunc, obj, integral)
