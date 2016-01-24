""" Modeling ordinary least squares
"""
import numpy as np
import numpy.linalg as npl

def fit(X, y):
    """ Do ordinary least squares fit of data Y to design matrix X

    Parameters
    ----------
    Y: array
       Data vector shape (N, ):
    X: array
       Design matrix 2D shape (N, P) where P is the number of parameters.

    Returns
    -------
    beta: array
          Coefficient vector (P, )
    """
    if X.ndim < 2:
        return (1. / X.dot(X)) * X.dot(y)
    beta = npl.pinv(X).dot(y)
    return beta