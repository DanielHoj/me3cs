import numpy as np

from me3cs.misc.handle_data import handle_zeros_in_scale


def savgol_coefficients(
    width: int, polyorder: int, deriv: int, delta: int
) -> np.ndarray:
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

    dim = data.ndim - 1
    if np.isscalar(constant):
        constant = np.asarray([constant for _ in range(data.shape[dim])])
    if np.isscalar(scale):
        scale = np.asarray([scale for _ in range(data.shape[dim])])

    scale = handle_zeros_in_scale(scale)
    return (data + constant) / scale
