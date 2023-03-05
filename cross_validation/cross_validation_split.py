from dataclasses import dataclass
from typing import Union

import numpy as np

from cross_validation.cross_validation_types import CrossValidationTypes


@dataclass
class CrossValidationSplit:
    x: np.ndarray
    y: np.ndarray = None

    percentage_left_out: float = None
    _cv_type: str = "venetian_blinds"
    n_splits: int = None

    test: Union[None, tuple[any]] = None
    training: Union[None, tuple[any]] = None

    @property
    def cv_type(self) -> str:
        return self._cv_type

    @cv_type.setter
    def cv_type(self, cv: str) -> None:
        cv_options = ["venetian_blinds", "contiguous_blocks", "random_blocks"]
        if cv not in cv_options:
            raise ValueError(f"Please input {cv_options}. {cv} was input")
        self.split(cv_type=cv)
        self._cv_type = cv

    def __post_init__(self) -> None:
        if self.n_splits is None:
            self.n_splits = int(1 / self.percentage_left_out)
        self.split(self.cv_type)

    def split(self, cv_type: str) -> None:
        cv = getattr(CrossValidationTypes, cv_type)
        x_training, x_test = cv(self.x, self.n_splits).subset()
        if self.y is not None:
            y_training, y_test = cv(self.y, self.n_splits).subset()
            test = tuple([x_test, y_test])
            training = tuple([x_training, y_training])
        else:
            test = tuple([x_test])
            training = tuple([x_training])

        self.n_splits = len(x_training)
        self.test = test
        self.training = training
