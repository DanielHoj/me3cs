import numpy as np


def transform_array_1d_to_2d(data: np.ndarray) -> np.ndarray:
    """
    Transform a 1D array into a 2D column vector. If the input array is already a
    2D column vector, it is returned unchanged.

    Parameters
    ----------
    data : numpy.ndarray
        The 1D or 2D column vector array to transform.

    Returns
    -------
    numpy.ndarray
        The transformed 2D column vector array.
    """
    if data.ndim == 1 or data.shape[1] == 1:
        return data.reshape(data.shape[0], 1)
    else:
        return data


def mask_arr(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Create a new array by masking a subset of elements from the input array.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    idx : numpy.ndarray
        The indices of the elements to mask.

    Returns
    -------
    numpy.ndarray
        The new array with the masked elements removed.
    """
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[idx] = False
    return arr[mask]


def handle_zeros_in_scale(
        scale: [np.ndarray | int | float], copy=True
) -> np.ndarray:
    """
    Replace zeros in a scaling array with a small constant to avoid division by zero.

    Parameters
    ----------
    scale : numpy.ndarray or int or float
        The scaling array or scalar value.
    copy : bool, optional
        Whether to make a copy of the scaling array, by default True.

    Returns
    -------
    numpy.ndarray
        The scaling array with small constants replacing the zeros.
    """
    if isinstance(scale, np.ndarray):
        constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
    elif isinstance(scale, (int, float)):
        if scale < 10 * np.finfo(float).eps:
            scale = 1.0
    return scale
