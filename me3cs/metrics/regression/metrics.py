import numpy as np

from me3cs.misc.metrics import rmse, bias, mse


class MetricsRegression:
    def __init__(self, y: np.ndarray, y_hat: np.ndarray):
        self.rmse = rmse(y, y_hat)
        self.mse = mse(y, y_hat)
        self.bias = bias(y, y_hat)
        self.variance = y_hat.std()

    def __repr__(self):
        cv_met = ", ".join(self.__dict__.keys())
        return f"Cross-validation metrics calculated:\n" \
               f"{cv_met}"
