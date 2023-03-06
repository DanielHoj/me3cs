import numpy as np


def interpolate_mean(data: np.ndarray) -> np.ndarray:
    return data


def interpolate_median(data: np.ndarray) -> np.ndarray:
    return data


interpolation_algorithms = {"mean": interpolate_mean,
                            "median": interpolate_median,
                            }
