from typing import TYPE_CHECKING

import numpy as np

from me3cs.cross_validation.cross_validation_model import CrossValidationModel
from me3cs.cross_validation.cross_validation_predictor import CrossValidationPredictor
from me3cs.cross_validation.cross_validation_preprocessing import (
    PreSplitPreprocessing,
    PreprocessingOnSplitData,
)
from me3cs.cross_validation.cross_validation_split import CrossValidationSplit
from me3cs.metrics.regression.metrics import MetricsRegression
from me3cs.preprocessing.called import Called

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationRegression:
    """
    Perform cross-validation on a regression model using the specified algorithm and number of components.

    Parameters:
    -----------
    x : np.ndarray
        The raw x data to be used for cross-validation.
    y : np.ndarray
        The raw y data to be used for cross-validation.
    called_preprocessing : [tuple[Called, Called], Called]
        The called preprocessing of the x and y data.
    algorithm : any
        The regression algorithm to use for cross-validation.
    n_components : int
        The number of components to use in the regression algorithm.
    cv_type : str
        The type of cross-validation to perform. Can be "loo" for Leave-One-Out, "lpo" for Leave-p-Out,
        or "kfold" for k-fold cross-validation.
    cv_metrics : any
        The type of metrics to use in cross-validation.
    percentage_left_out : float, optional
        The percentage of data to leave out for cross-validation. Defaults to 0.1.

    Attributes:
    -----------
    x : np.ndarray
        The x data used for cross-validation.
    y : np.ndarray
        The y data used for cross-validation.
    called_preprocessing : [tuple[Called, Called], Called]
        The called preprocessing of the x and y data.
    percentage_left_out : float
        The percentage of data to leave out for cross-validation.
    algorithm : any
        The regression algorithm used for cross-validation.
    n_components : int
        The number of components used in the regression algorithm.
    cv_type : str
        The type of cross-validation performed. Can be "loo" for Leave-One-Out, "lpo" for Leave-p-Out,
        or "kfold" for k-fold cross-validation.
    cv_metrics : any
        The type of metrics used in cross-validation.
    results : any
        The results of the cross-validation.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            called_preprocessing: [tuple[Called, Called], Called],
            algorithm: "TYPING_ALGORITHM_REGRESSION",
            n_components: int,
            cv_type: str,
            cv_metrics: MetricsRegression,
            percentage_left_out: float = 0.1,
    ) -> None:

        self.x = x
        self.y = y
        self.called_preprocessing = called_preprocessing
        self.percentage_left_out = percentage_left_out
        self.algorithm = algorithm
        self.n_components = n_components
        self.cv_type = cv_type
        self.cv_metrics = cv_metrics
        self.results = None
        self.fit()

    def fit(self) -> None:
        if self.cv_type is None:
            return

        x_called, y_called = self.called_preprocessing
        # Preprocess with non scaling methods:
        partly_preprocessed_x = PreSplitPreprocessing(
            data=self.x, called=x_called
        ).data

        # Split data based on the cross-validation type:
        split = CrossValidationSplit(
            x=partly_preprocessed_x,
            y=self.y,
            percentage_left_out=self.percentage_left_out,
            cv_type=self.cv_type,
        )

        #
        preprocessed_split = PreprocessingOnSplitData(
            split=split, x_called=x_called, y_called=y_called
        )
        test_set = preprocessed_split.test_set
        training_set = preprocessed_split.training_set

        models = CrossValidationModel(
            algorithm=self.algorithm,
            n_components=self.n_components,
            training=training_set,
        )

        predictor = CrossValidationPredictor(test_set=test_set, models=models.cv_models)
        y_test = np.concatenate(test_set[1])
        self.results = MetricsRegression(y_test, predictor.predictor_results)


