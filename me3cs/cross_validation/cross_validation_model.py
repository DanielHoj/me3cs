from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationModel:
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
