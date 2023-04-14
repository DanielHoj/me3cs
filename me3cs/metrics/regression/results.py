import numpy as np

from me3cs.framework.helper_classes.options import dict_to_string_with_newline
from me3cs.misc.metrics import explained_variance, rmse, mse, bias
from me3cs.models.regression.mlr import MLR
from me3cs.models.regression.pcr import PCR
from me3cs.models.regression.pls import SIMPLS, NIPALS


class ResultsRegression:
    def __repr__(self):
        return f"Calibration metrics calculated:\n" \
               f"{dict_to_string_with_newline(self.__dict__)}"


class ResultsPLS(ResultsRegression):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            results: [SIMPLS, NIPALS],
    ):
        self.reg = results.reg
        self.y_hat = np.dot(x, results.reg)
        self.rmse = rmse(y, self.y_hat)
        self.mse = mse(y, self.y_hat)
        self.bias = bias(y, self.y_hat)
        self.variance = self.y_hat.std()

        self.x_scores = results.x_scores
        self.x_loadings = results.x_loadings
        self.x_weight = results.x_weight
        self.y_scores = results.y_scores
        self.y_loadings = results.y_loadings

        self.explained_var_x = explained_variance(results.x_loadings, x.shape[0])
        self.explained_var_y = explained_variance(results.y_loadings, x.shape[0])
        self.cum_explained_var_x = np.cumsum(self.explained_var_x)


class ResultsMLR:
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            results: MLR,
    ):
        self.__dict__.update(results.__dict__)
        self.y_hat = np.dot(x, results.reg)
        self.rmse = rmse(y, self.y_hat)
        self.mse = mse(y, self.y_hat)
        self.bias = bias(y, self.y_hat)
        self.variance = self.y_hat.std()


class ResultsPCR:
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            results: PCR,
    ):
        self.__dict__.update(results.__dict__)

        self.y_hat = np.dot(x, results.reg)
        self.rmse = rmse(y, self.y_hat)
        self.mse = mse(y, self.y_hat)
        self.bias = bias(y, self.y_hat)


class ResultsSVM:
    pass


RegressionResults = {
    "PLS": ResultsPLS,
    "MLR": ResultsMLR,
    "PCR": ResultsPCR,
    "SVM": ResultsSVM,
}
