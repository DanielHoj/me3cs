import numpy as np

from me3cs.misc.metrics import rmse, bias, mse


class MetricsRegression:
    def __init__(self, y: np.ndarray, y_hat: np.ndarray):
        self.y_hat = y_hat
        self.rmse = rmse(y, self.y_hat)
        self.mse = mse(y, self.y_hat)
        self.bias = bias(y, self.y_hat)
        self.variance = self.y_hat.std()
