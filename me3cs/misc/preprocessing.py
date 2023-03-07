import numpy as np

from me3cs.misc.handle_data import handle_zeros_in_scale


def savgol_coefficients(
    width: int, polyorder: int, deriv: int, delta: int
) -> np.ndarray:
    """
    Computes Savitzky-Golay filter coefficients for the given parameters.

    Parameters
    ----------
    width : int
        The window size of the filter.
    polyorder : int
        The order of the polynomial to fit to the data.
    deriv : int
        The order of the derivative to compute.
    delta : int
        The spacing of the data points.

    Returns
    -------
    np.ndarray
        The filter coefficients of shape (polyorder+1,).

    Notes
    -----
    The coefficients are computed using the least-squares method on the matrix
    equation A*c = y, where A is a matrix of shape (width, polyorder+1) containing
    the powers of x, and y is a vector of length polyorder+1 containing the desired
    derivative coefficients.

    References
    ----------
    .. [1] Savitzky, A., Golay, M. J. E. (1964). Smoothing and Differentiation of Data by
           Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627–1639.
    """
    halflen, rem = divmod(width, 2)
    x = np.arange(-halflen, width - halflen, dtype=float)

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x**order

    y = np.zeros(polyorder + 1)

    y[deriv] = np.math.factorial(deriv) / (delta**deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def preprocessing_scaling(
    data: np.ndarray, constant: np.ndarray, scale: [np.ndarray, float]
) -> np.ndarray:
    """
    Scale the input data and center it.

    Parameters
    ----------
    data : numpy.ndarray
        Input data to scale and center.
    constant : numpy.ndarray
        Array of constants used to center the data. If a scalar is provided, it will be
        broadcasted to match the number of features in the data.
    scale : numpy.ndarray or float
        Array of scaling factors used to scale the data. If a scalar is provided, it will be
        broadcasted to match the number of features in the data.

    Returns
    -------
    numpy.ndarray
        Scaled and centered data.

    Notes
    -----
    This function handles zeros in the scale array by replacing them with a small non-zero value.

    """
    dim = data.ndim - 1
    if np.isscalar(constant):
        constant = np.asarray([constant for _ in range(data.shape[dim])])
    if np.isscalar(scale):
        scale = np.asarray([scale for _ in range(data.shape[dim])])

    scale = handle_zeros_in_scale(scale)
    return (data + constant) / scale
