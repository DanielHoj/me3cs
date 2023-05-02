from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationPredictor:
    """
    Predicts the target values for the test set using the trained models in a cross-validation setting.

    Parameters
    ----------
    test_set : tuple[..., np.ndarray]
        The test set input data.
    models : list[..., "TYPING_ALGORITHM_REGRESSION"]
        The list of trained regression models.

    Attributes
    ----------
    test_set : tuple[..., np.ndarray]
        The test set input data.
    models : list[..., "TYPING_ALGORITHM_REGRESSION"]
        The list of trained regression models.
    predictor_results : [None, np.ndarray]
        The concatenated predictions of the target values for the test set.
    """
    def __init__(self, test_set: tuple[..., np.ndarray],
                 models: list[..., "TYPING_ALGORITHM_REGRESSION"],
                 ) -> None:

        self.test_set = test_set
        self.models = models

        self.predictor_results: [None, np.ndarray] = None
        self.predictor()

    def predictor(self) -> None:
        """
        Computes the predictions for the test set using the trained models.
        """
        predictor_results = list()
        x_test = self.test_set[0]
        for i, model in enumerate(self.models):
            reg = model.reg
            predictor_results.append(np.dot(x_test[i], reg))

        self.predictor_results = np.concatenate(predictor_results)
