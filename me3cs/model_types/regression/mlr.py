# TODO: create mlr algorithm class
import numpy as np

from me3cs.misc.metrics import moore_penrose_inverse


class MLR:
    def __init__(self, x: np.ndarray, y: np.ndarray, n_components=None) -> None:
        self.x = x
        self.y = y
        self.reg = None
        self.fit()

    def fit(self) -> None:
        x = self.x
        y = self.y
        self.reg = np.dot(moore_penrose_inverse(x), y)
