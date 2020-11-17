# hierScale
### Hussein Hazimeh and Rahul Mazumder
### Massachusetts Institute of Technology

## Introduction
hierScale is scalable toolkit for fitting sparse linear regression models with pairwise feature interactions. The optimization is done under the **strong hierarchy (SH)** constraint: an interaction coefficient is non-zero only if its associated main feature coefficients are non-zero. This constraint can enhance the interpretability of sparse interaction models and also reduce the future data collection costs; see the discussion in [(Hazimeh and Mazumder, 2020)](http://proceedings.mlr.press/v108/hazimeh20a.html).

More formally, given a data matrix X of main features and a response vector y, the toolkit fits a **convex relaxation** of the following model:

<img src="https://raw.githubusercontent.com/hazimehh/hierScale/master/formulation.png" width = 450>

 where X_i denotes the ith column (feature) of X and * refers to element-wise multiplication. The L0 norms impose sparsity on the coefficients and the constraints enforce SH. See [(Hazimeh and Mazumder, 2020)](http://proceedings.mlr.press/v108/hazimeh20a.html) for details on how the convex relaxation of the above problem is derived. The optimization is done for a regularization path (i.e., over a grid of lambda_1's and lambda_2's). We use proximal gradient descent (PGD) for optimization, along with novel proximal screening and gradient screening rules, which speed up PGD by over 4900x.

## Installation
hierScale is written in Python 3. It uses Gurobi internally (for solving the LPs required for checking the optimality conditions). Before installing hierScale, please make sure that  [Gurobi](https://www.gurobi.com) and its Python interface (gurobipy) are installed.

To install hierScale, run the following command:
```bash
pip install hierScale
```

## Quick Start
In Python, assuming you have the data X and y stored as numpy arrays, run the following to fit a regularization path:
```python
from hierScale import hier_fit, hier_predict

# Set the parameters of the algorithm.
params = {
    "nLambda": 100, # Number of lambda_1's in the path.
    "maxSuppSize": 500, # Max support size to terminate the path at.
}

# Fit a path.
solutions, lambdas = hier_fit(X, y, params)

# solutions is a list of all the solutions in the path.
# To access the ith solution, say i=10, use the following:
current_solution = solutions[10]
current_solution.B # A dictionary of the non-zero coefficients in beta.
current_solution.T # A dictionary of the non-zero coefficients in theta.
current_solution.intercept # The intercept term.

# To predict the response given a matrix X, run the following:
hier_predict(current_solution, X)

# For more advanced usage and parameters, please check the documentation:
print(hier_fit.__doc__)
```





## References
[Learning Hierarchical Interactions at Scale: A Convex Optimization Approach](http://proceedings.mlr.press/v108/hazimeh20a.html).

Bibtex citation below:
```
@InProceedings{pmlr-v108-hazimeh20a, 
title = {Learning Hierarchical Interactions at Scale: A Convex Optimization Approach},
author = {Hazimeh, Hussein and Mazumder, Rahul},
booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
pages = {1833--1843},
year = {2020},
editor = {Silvia Chiappa and Roberto Calandra},
volume = {108},
series = {Proceedings of Machine Learning Research},
address = {Online},
month = {26--28 Aug},
publisher = {PMLR},
pdf = {http://proceedings.mlr.press/v108/hazimeh20a/hazimeh20a.pdf},
url = {http://proceedings.mlr.press/v108/hazimeh20a.html}}
```
