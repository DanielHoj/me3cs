import numpy as np

from me3cs.misc.metrics import rmse, bias, mse


class MetricsRegression:
    """
    Class to calculate the regression metrics for evaluating regression models.

    Parameters
    ----------
    y : np.ndarray
        The true output values.
    y_hat : np.ndarray
        The predicted output values.

    Attributes
    ----------
    rmse : float
        The root-mean-square error (RMSE) of the regression model.
    mse : float
        The mean squared error (MSE) of the regression model.
    bias : float
        The bias of the regression model.
    variance : float
        The variance of the predicted output values (y_hat).
        """
    def __init__(self, y: np.ndarray, y_hat: np.ndarray):
        self.rmse = rmse(y, y_hat)
        self.mse = mse(y, y_hat)
        self.bias = bias(y, y_hat)
        self.variance = y_hat.std()

    def __repr__(self):
        cv_met = ", ".join(self.__dict__.keys())
        return f"Cross-validation metrics calculated:\n" \
               f"{cv_met}"
