import numpy as np


def transform_array_1d_to_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1 or data.shape[1] == 1:
        return data.reshape(data.shape[0], 1)
    else:
        return data


def mask_arr(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[idx] = False
    return arr[mask]


def handle_zeros_in_scale(
        scale: [np.ndarray | int | float], copy=True
) -> np.ndarray:
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
