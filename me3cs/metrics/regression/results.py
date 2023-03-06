import numpy as np

from me3cs.misc.metrics import latent_variable, explained_variance, rmse, mse, bias, leverage
from me3cs.model_types.regression.mlr import MLR
from me3cs.model_types.regression.pcr import PCR
from me3cs.model_types.regression.pls import SIMPLS, NIPALS


class ResultsPLS:
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            results: [SIMPLS, NIPALS],
    ):
        self.__dict__.update(results.__dict__)

        self.y_hat = np.dot(x, results.reg)
        self.rmse = rmse(y, self.y_hat)
        self.mse = mse(y, self.y_hat)
        self.bias = bias(y, self.y_hat)
        self.variance = self.y_hat.std()

        self.latent_variable = latent_variable(results.x_scores, results.x_loadings)
        self.leverage = leverage(results.x_scores, x.shape[0])
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
