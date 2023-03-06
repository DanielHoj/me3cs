from typing import Tuple

import numpy as np
import scipy.stats as st

from me3cs.misc.handle_data import handle_zeros_in_scale


def normalise(data: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(data)
    return np.divide(data, handle_zeros_in_scale(norm))


def moore_penrose_inverse(data: np.ndarray) -> np.ndarray:
    if data.shape[0] > data.shape[1]:
        results = np.linalg.inv(np.dot(data.T, data)) @ data.T
    elif data.shape[0] < data.shape[1]:
        results = data.T @ np.linalg.inv(np.dot(data, data.T))
    else:
        results = np.linalg.inv(data)
    return results


def rmse(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(actual - predicted), axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(np.square(actual - predicted), axis=axis)


def bias(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sum((predicted - actual) / actual.shape[0], axis=axis)


def residuals(x: np.ndarray, scores: np.ndarray, loadings: np.ndarray, axis: int = 2) -> np.ndarray:
    x_repeated = np.broadcast_to(x[:, :, np.newaxis], (x.shape[0], x.shape[1], scores.shape[1]))
    results = x_repeated - np.einsum('ji,ki->jki', scores, loadings).cumsum(axis=axis)
    return results


def t_crit(data: np.ndarray, confidence_limit: float = 0.95) -> np.ndarray:
    dof = len(data)
    return np.abs(st.t.ppf((1 - confidence_limit) / 2, dof))


def confidence_interval(array: np.ndarray) -> Tuple[float, float]:
    return st.t.interval(0.95, len(array) - 1, loc=np.mean(array), scale=st.sem(array))


def confidence_limit_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.empty(matrix.shape[-1])
    for i in range(matrix.shape[-1]):
        result[i] = confidence_interval(matrix[:, i])[1]
    return result


def q_residuals(residual_matrix: np.ndarray) -> np.ndarray:
    results = np.einsum('ijk,ijk->ik', residual_matrix, residual_matrix)
    return results


def latent_variable(scores: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    results = np.einsum("ji, ki -> jki", scores, loadings)
    return results


def explained_variance(loadings: np.ndarray, n: np.ndarray) -> np.ndarray:
    results = np.diag(loadings.T @ loadings) / (n - 1)
    return results


def leverage(scores: np.ndarray, n: int) -> np.ndarray:
    results = np.diag(scores @ scores.T) + (1 / n)
    return results
