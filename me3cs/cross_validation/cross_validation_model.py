from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationModel:
    """
    Creates a list of models by fitting the specified regression algorithm on the training data.

    Parameters
    ----------
    algorithm : TYPING_ALGORITHM_REGRESSION
        The regression algorithm to use for cross-validation.
    n_components : int
        The number of components to use in the regression algorithm.
    training : [tuple[list[np.ndarray, ...]], tuple[list[np.ndarray, ...], list[np.ndarray, ...]]]
        Tuple containing the lists of preprocessed training input data (x_training) and output
        data (y_training).

    Attributes
    ----------
    algorithm : TYPING_ALGORITHM_REGRESSION
        The regression algorithm to use for cross-validation.
    n_components : int
        The number of components to use in the regression algorithm.
    training : [tuple[list[np.ndarray, ...]], tuple[list[np.ndarray, ...], list[np.ndarray, ...]]]
        Tuple containing the lists of preprocessed training input data (x_training) and output
        data (y_training).
    cv_models : [None, list[..., "TYPING_ALGORITHM_REGRESSION"]]
        List of trained regression models for each fold in cross-validation.
    """
    def __init__(self,
                 algorithm: "TYPING_ALGORITHM_REGRESSION",
                 n_components: int,
                 training: [
                     tuple[list[np.ndarray, ...]],
                     tuple[list[np.ndarray, ...], list[np.ndarray, ...]],
                 ]) -> None:
        self.algorithm = algorithm
        self.n_components = n_components
        self.training = training

        self.cv_models: [None, list[..., "TYPING_ALGORITHM_REGRESSION"]] = None

        self.fit()

    def fit(self) -> None:
        """
        Fits the models on the training data using the specified regression algorithm.

        Trains the regression algorithm on each fold of the training data and stores
        the resulting models in the cv_models attribute.
        """
        algorithm = self.algorithm
        x_training, y_training = self.training
        n_splits = len(x_training)
        self.cv_models = [
            algorithm(
                x=x_training[i], y=y_training[i], n_components=self.n_components
            )
            for i in range(n_splits)
            if print(f"Model {i + 1} of {n_splits}") or True
        ]
