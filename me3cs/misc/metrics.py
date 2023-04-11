import numpy as np
import scipy.stats as st

from me3cs.misc.handle_data import handle_zeros_in_scale


def normalise(data: np.ndarray) -> np.ndarray:
    """
    Normalise the input data using L2 norm.

    Parameters
    ----------
    data : np.ndarray
        The input data.

    Returns
    -------
    np.ndarray
        The normalised input data.

    """
    norm = np.linalg.norm(data)
    return np.divide(data, handle_zeros_in_scale(norm))


def moore_penrose_inverse(data: np.ndarray) -> np.ndarray:
    """
    Compute the Moore-Penrose inverse of the input data. Full rank is assumed.

    Parameters
    ----------
    data : np.ndarray
        The input data.

    Returns
    -------
    np.ndarray
        The Moore-Penrose inverse of the input data.

    """
    if data.shape[0] > data.shape[1]:
        results = np.linalg.inv(np.dot(data.T, data)) @ data.T
    elif data.shape[0] < data.shape[1]:
        results = data.T @ np.linalg.inv(np.dot(data, data.T))
    else:
        results = np.linalg.inv(data)
    return results


def rmse(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the root mean squared error between actual and predicted values.

    Parameters
    ----------
    actual : np.ndarray
        The actual values.
    predicted : np.ndarray
        The predicted values.
    axis : int, optional
        The axis along which to compute the RMSE, by default 0.

    Returns
    -------
    np.ndarray
        The root mean squared error between actual and predicted values.
    """
    return np.sqrt(np.mean(np.square(actual - predicted), axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the mean squared error between actual and predicted values.

    Parameters
    ----------
    actual : np.ndarray
       The actual values.
    predicted : np.ndarray
       The predicted values.
    axis : int, optional
       The axis along which to compute the MSE, by default 0.

    Returns
    -------
    np.ndarray
       The mean squared error between actual and predicted values.

    """
    return np.mean(np.square(actual - predicted), axis=axis)


def bias(actual: np.ndarray, predicted: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the bias between actual and predicted values.

    Parameters
    ----------
    actual : np.ndarray
        The actual values.
    predicted : np.ndarray
        The predicted values.
    axis : int, optional
        The axis along which to compute the bias, by default 0.

    Returns
    -------
    np.ndarray
        The bias between actual and predicted values.

    """
    return np.sum((predicted - actual) / actual.shape[0], axis=axis)


def residuals(x: np.ndarray, scores: np.ndarray, loadings: np.ndarray, axis: int = 2) -> np.ndarray:
    """
    Compute the residuals between the actual data and the predicted data.

    Parameters
    ----------
    x : np.ndarray
        The input data.
    scores : np.ndarray
        The scores obtained from PCA.
    loadings : np.ndarray
        The loadings obtained from PCA.
    axis : int, optional
        The axis along which to compute the residuals, by default 2.

    Returns
    -------
    np.ndarray
        The residuals between the actual data and the predicted data.
    """
    x_repeated = np.broadcast_to(x[:, :, np.newaxis], (x.shape[0], x.shape[1], scores.shape[1]))
    results = x_repeated - np.einsum('ji,ki->jki', scores, loadings).cumsum(axis=axis)
    return results


def t_crit(data: np.ndarray, confidence_limit: float = 0.95) -> np.ndarray:
    """
    Computes the critical value of the t-distribution for a given confidence level and degrees of freedom.

    Parameters
    ----------
    data : np.ndarray
        The data for which the critical value is computed.
    confidence_limit : float, optional (default=0.95)
        The confidence level for which the critical value is computed.

    Returns
    -------
    np.ndarray
        The critical value of the t-distribution for the given confidence level and degrees of freedom.
    """
    dof = len(data)
    return np.abs(st.t.ppf((1 - confidence_limit) / 2, dof))


def confidence_interval(array: np.ndarray) -> tuple[float, float]:
    """
    Computes the confidence interval for a given array of data using the t-distribution.

    Parameters
    ----------
    array : np.ndarray
        The array of data for which the confidence interval is computed.

    Returns
    -------
    tuple[float, float]
        The lower and upper bounds of the confidence interval.
    """
    return st.t.interval(0.95, len(array) - 1, loc=np.mean(array), scale=st.sem(array))


def confidence_limit_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Computes the upper bound of the confidence interval for each column of a matrix of data using the t-distribution.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix of data for which the upper bounds of the confidence intervals are computed.

    Returns
    -------
    np.ndarray
        An array containing the upper bounds of the confidence intervals for each column of the matrix.
    """
    result = np.empty(matrix.shape[-1])
    for i in range(matrix.shape[-1]):
        result[i] = confidence_interval(matrix[:, i])[1]
    return result


def q_residuals(residual_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the Q-residuals for a given residual matrix.

    Parameters
    ----------
    residual_matrix : np.ndarray
        The residual matrix for which the Q-residuals are computed.

    Returns
    -------
    np.ndarray
        An array containing the Q-residuals for each observation in the residual matrix.
    """
    results = np.einsum('ijk,ijk->ik', residual_matrix, residual_matrix)
    return results


def latent_variable(scores: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """
    Computes the estimated matrix of data from a given set of scores and loadings.

    Parameters
    ----------
    scores : np.ndarray
        The scores matrix.
    loadings : np.ndarray
        The loadings matrix.

    Returns
    -------
    np.ndarray
        The estimated matrix of data.
    """
    results = np.einsum("ji, ki -> jki", scores, loadings)
    return results


def explained_variance(loadings: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Computes the explained variance for each column of a loadings matrix.

    Parameters
    ----------
    loadings : np.ndarray
        The loadings matrix.
    n : np.ndarray
        The number of observations.

    Returns
    -------
    np.ndarray
        An array containing the explained variance for each column of the loadings matrix.
    """
    results = np.diag(loadings.T @ loadings) / (n - 1)
    return results


def leverage(scores: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the leverage values for a given set of scores.

    Parameters
    ----------
    scores : np.ndarray
        The scores matrix of shape (n_samples, n_components).
    n : int
        The number of samples in the original dataset.

    Returns
    -------
    np.ndarray
        The leverage values of shape (n_samples,).

    Notes
    -----
    The leverage values indicate the contribution of each sample to the principal
    components. High leverage values indicate that the sample has a large impact
    on the principal components and may be influential in the analysis.
    """
    results = np.diag(scores @ scores.T) + (1 / n)
    return results


def hotellings_t2(scores):
    results = [(np.diag(scores[:, :i] @ scores[:, :i].T) + (1 / scores.shape[1])).reshape(-1, 1) for i in
               range(1, scores.shape[1] + 1)]
    return np.concatenate(results, axis=1)


def calculate_vips(scores, weights, loadings):
    """
    Calculates VIP scores for variables in X based on scores T and weights W from a PLS model.

    Parameters
    ----------
    scores: numpy.ndarray
        An n x m array of PLS scores.
    weights: numpy.ndarray
        An m x p array of PLS weights.
    loadings: numpy.ndarray
        An m x p array of PLS loadings.

    Returns
    -------
    vip_scores (numpy.ndarray): A 1D array of VIP scores for each variable in X.
    """
    p, h = weights.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(scores.T, scores), loadings.T), loadings))
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(weights[i, j] / np.linalg.norm(weights[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p*(np.matmul(s.T, weight))/total_s)

    return vips


def calculate_vips2(x_scores, x_weights, y_loadings):
    """
    Calculates VIP scores for variables in X based on scores T and weights W from a PLS model.

    Parameters
    ----------
    x_scores: numpy.ndarray
        An n x m array of x scores from a PLS model.
    x_weights: numpy.ndarray
        An m x p array of x weights from a PLS model.
    y_loadings: numpy.ndarray
        An m x p array of y loadings from a PLS model.

    Returns
    -------
    vip_scores (numpy.ndarray): A 2D array of VIP scores for each variable in X, for each component.
    """

    k, a = x_weights.shape
    s = np.sum(x_scores ** 2, axis=0) * np.sum(y_loadings ** 2, axis=0)
    total_s = np.cumsum(s)

    weights_norm = x_weights / np.linalg.norm(x_weights, axis=0)
    weights_squared = weights_norm**2

    vip = np.sqrt(k/a * np.cumsum(s * weights_squared, axis=1) / total_s)

    return vip
