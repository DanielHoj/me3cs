import numpy as np


class Attributes:
    """
    Sets dimension and shape of data.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.dimensions = self.set_dimensions(data)
        self.shape = self.set_shape(data)
        self.data = data

    @staticmethod
    def set_dimensions(data: np.ndarray) -> int:
        return data.ndim

    @staticmethod
    def set_shape(data: np.ndarray) -> tuple:
        return data.shape
