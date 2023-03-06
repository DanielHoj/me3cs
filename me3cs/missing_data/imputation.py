from functools import partial

import numpy as np
from scipy.sparse.linalg import svds

from me3cs.misc.handle_data import handle_zeros_in_scale


def emsvd(data, k=None, tol=1E-3, maxiter=None):
    """
    Approximate SVD on data with missing values via expectation-maximization

    Inputs:
    -----------
    Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
    k:          number of singular values/vectors to find (default: k=ndim)
    tol:        convergence tolerance on change in trace norm
    maxiter:    maximum number of EM steps to perform (default: no limit)

    Returns:
    -----------
    Y_hat:      (nobs, ndim) reconstructed data matrix
    mu_hat:     (ndim,) estimated column means for reconstructed data
    U, s, Vt:   singular values and vectors (see np.linalg.svd and
                scipy.sparse.linalg.svds for details)
    """

    if k is None:
        svdmethod = partial(np.linalg.svd, full_matrices=False)
    else:
        svdmethod = partial(svds, k=k)
    if maxiter is None:
        maxiter = np.inf

    # initialize the missing values to their respective column means
    mu_hat = np.nanmean(data, axis=0, keepdims=True)
    valid = np.isfinite(data)
    data_hat = np.where(valid, data, mu_hat)

    halt = False
    ii = 0
    v_prev = 0

    while not halt:
        ii += 1
        # SVD on filled-in data
        U, s, Vt = svdmethod(data_hat - mu_hat)

        # impute missing values
        data_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

        # update bias parameter
        mu_hat = data_hat.mean(axis=0, keepdims=1)

        # test convergence using relative change in trace norm
        v = s.sum()
        if ii >= maxiter or ((v - v_prev) / handle_zeros_in_scale(v_prev)) < tol:
            halt = True
        v_prev = v

    return data_hat


def nipals_missing_data(data: np.ndarray) -> np.ndarray:
    return data


imputation_algorithms = {"emsvd": emsvd,
                         "nipals": nipals_missing_data,
                         }
