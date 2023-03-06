from dataclasses import dataclass

import numpy as np


@dataclass
class CrossValidationPredictor:
    test_set: tuple[any]
    models: list[any]
    predictor_results = None

    def __post_init__(self):
        self.predictor()

    def predictor(self):
        self.predictor_results = list()
        test_set = self.test_set
        if len(test_set) > 0:
            x_test = test_set[0]
        else:
            x_test, y_test = test_set
        for i, model in enumerate(self.models):
            reg = model.reg
            self.predictor_results.append(np.dot(x_test[i], reg))
        self.predictor_results = np.concatenate(self.predictor_results)
