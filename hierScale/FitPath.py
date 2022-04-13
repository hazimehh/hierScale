"""Module for fitting a regularization path."""

import time
from copy import deepcopy
from .InitialRun import InitialRun
from .Structs import *
from .FitSingleLambda import FitSingleLambda

def hier_fit(X, y, params):
    """
    Fits a regularization path for the interactions model under strong hierarchy.
    The grid of lambda_1's and lambda_2's is selected automatically.

    Inputs:
        X: Data matrix (numpy array) cotaining the main features. Interactions columns are
          generated on-the-fly and should not be included in X.
        y: Response vector (numpy array).
        params: A dictionary of parameters that control the fitting procedure.
          params supports the following options:
            "M": Float. Adds box constraints of the form |beta_i| <= M, |theta_{ij}| <= M.
              Default is 10^10.
            "alpha": Float. Used to define lambda_2: lambda_2 = alpha * lambda_1. Larger alphas
              lead to more sparsity in the interaction terms. To avoid performance issues,
              we recommend alpha >=1. Default is 2.
            "tol": Tolerance for PGD. Default 1e-6.
            "tolCD": Tolerance for Block Coordinate Ascent. Default is 1e-6.
            "nLambda": Number of solutions in the path. Default is 100.
            "lambdaMinRatio": A float < 1. The path is terminated at
              lambdaMinRatio*lambda_1_max, where lambda_1_max is the first lambda_1
              in the path (which sets all the main effects to zero). Smaller values
              will lead to denser solutions in the path. Default is 0.05.
            "maxSuppSize": The maximum number of non-zeros in the path after which to
              terminate.

    Outputs:
      A tuple (solutions, lambdas).
        solutions: A list of solutions. solutions[i] corresponds to the ith lambda
          in the path. solutions[i].B is a dictionary of the non-zero elements in
          B (the main coefficients vector) and solutions[i].T is a dictionary of the
          non-zero elements in T (the interaction effects vector). The intecept can
          be accessed as follows: solutions[i].intercept.
        lambdas: A list of the lambda_1's used in the path.
    """
    data = Matrices(X, y)
    Params = deepcopy(params)
    # Initialize the parameters not provided by the user.
    Params = InitializeParams(Params)
    # Store the print function in Params.
    Params["print"] = print if Params["debug"] else lambda *a, **k: None
    start = time.time()
    kmain = 5
    kint = 5
    # Get the initial active set and lambda_1 max.
    BActive, TActive, lambdaMax, maxNorm = InitialRun(
        data, kmain, kint, Params)
    # Generate the seq. of lambda_1's to be used in the path.
    Lambdas = np.geomspace(
        lambdaMax,
        Params["lambdaMinRatio"] *
        lambdaMax,
        num=Params["nLambda"])
    data.maxNorm = maxNorm  # used by CheckOptScreen.
    data.Augment(BActive, TActive)
    fixedVars = {}
    initialSol = Solution([], [])  # Change later when we start using this.
    u = {}
    w = {}
    useDual = False
    relaxationVars = RelaxationVars(
        BActive, TActive, initialSol, useDual, u, w)
    solutions = []
    for i in range(Params["nLambda"]):  # nLambda
        print("Path Iteration: ", i)
        Params["print"]("Lambda = ", Lambdas[i])
        Params["Lambda"] = Lambdas[i]
        # relaxationVars is updated by FitSingleLambda.
        sol, obj, integral = FitSingleLambda(
            Params, relaxationVars, fixedVars, data)
        solutions.append(deepcopy(sol))
        nnz = len(sol.B) + len(sol.T)
        if nnz >= Params["maxSuppSize"]:
            break
        Params["print"]("Number of nnz: ", nnz)
    end = time.time()
    Params["print"]("Total Elapsed Time: ", end - start)
    return solutions, Lambdas
