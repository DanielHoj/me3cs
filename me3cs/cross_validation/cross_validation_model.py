from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class CrossValidationModel:
    algorithm: any
    n_components: int
    training: Union[
        tuple[list[np.ndarray, ...]],
        tuple[list[np.ndarray, ...], list[np.ndarray, ...]],
    ]

    cv_models: List[any] = None

    def __post_init__(self) -> None:
        self.fit()

    def fit(self) -> None:
        algorithm = self.algorithm

        if len(self.training) > 1:
            x_training, y_training = self.training
            n_splits = len(x_training)
            self.cv_models = [
                algorithm(
                    x=x_training[i], y=y_training[i], n_components=self.n_components
                )
                for i in range(n_splits)
                if print(f"Model {i+1} of {n_splits}") or True
            ]
        else:
            x_training = self.training
            n_splits = len(x_training)
            self.cv_models = [
                algorithm(x=x_training[i], n_components=self.n_components)
                for i in range(n_splits)
                if print(f"Model {i+1} of {n_splits}") or True
            ]
