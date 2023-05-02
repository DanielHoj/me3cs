from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationPredictor:
    def __init__(self, test_set: tuple[..., np.ndarray],
                 models: list[..., "TYPING_ALGORITHM_REGRESSION"],
                 ) -> None:

        self.test_set = test_set
        self.models = models

        self.predictor_results: [None, np.ndarray] = None
        self.predictor()

    def predictor(self) -> None:
        predictor_results = list()
        x_test = self.test_set[0]
        for i, model in enumerate(self.models):
            reg = model.reg
            predictor_results.append(np.dot(x_test[i], reg))

        self.predictor_results = np.concatenate(predictor_results)
