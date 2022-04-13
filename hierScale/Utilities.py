"""Helper Utilities. Only contains the Predict function for now."""

import numpy as np


def hier_predict(sol, X):
    '''
    A predict function for hierScale.
    Inputs:
        sol: A single solution obtained from hier_fit.
        X: Matrix of (main effects) to predict the response.
    Output: A predictions vector.
    '''
    BIndices = sorted(sol.B.keys())
    TIndices = sorted(sol.T.keys())
    B, T = sol.ToArray(BIndices, TIndices)
    if len(TIndices) != 0:
        XTnew = np.hstack([(X[:, i] * X[:, j])[:, np.newaxis]
                           for (i, j) in TIndices])
        TPred = np.dot(XTnew, T)
    else:
        TPred = 0
    return np.dot(X[:, BIndices], B) + TPred + sol.intercept
