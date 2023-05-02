from typing import TYPE_CHECKING

import numpy as np

from .cross_validation_model import CrossValidationModel
from .cross_validation_predictor import CrossValidationPredictor
from .cross_validation_preprocessing import PreSplitPreprocessing, PreprocessingOnSplitData
from .cross_validation_split import CrossValidationSplit

from me3cs.metrics.regression.metrics import MetricsRegression
from me3cs.preprocessing.called import Called

if TYPE_CHECKING:
    from me3cs.framework.regression_model import TYPING_ALGORITHM_REGRESSION


class CrossValidationRegression:
    """
    Performs cross-validation for a given regression algorithm on provided data.

    Parameters
    ----------
    x : np.ndarray
        Input feature matrix (n_samples, n_features).
    y : np.ndarray
        Output target array (n_samples,).
    called_preprocessing : [tuple[Called, Called], Called]
        Preprocessing functions to apply on the input data before cross-validation.
    algorithm : TYPING_ALGORITHM_REGRESSION
        The regression algorithm to use for cross-validation.
    n_components : int
        The number of components to use in the regression algorithm.
    cv_type : str
        The type of cross-validation to perform.
    cv_metrics : MetricsRegression
        Metrics to evaluate the performance of the regression model.
    percentage_left_out : float, optional, default=0.1
        The percentage of data to leave out for validation during cross-validation.

    Attributes
    ----------
    x : np.ndarray
        Input feature matrix (n_samples, n_features).
    y : np.ndarray
        Output target array (n_samples,).
    called_preprocessing : [tuple[Called, Called], Called]
        Preprocessing functions to apply on the input data before cross-validation.
    percentage_left_out : float
        The percentage of data to leave out for validation during cross-validation.
    algorithm : TYPING_ALGORITHM_REGRESSION
        The regression algorithm to use for cross-validation.
    n_components : int
        The number of components to use in the regression algorithm.
    cv_type : str
        The type of cross-validation to perform.
    cv_metrics : MetricsRegression
        Metrics to evaluate the performance of the regression model.
    results : MetricsRegression or None
        The performance metrics of the fitted model, or None if the model is not yet fitted.
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
        """
        Fits the cross-validation model on the provided data.

        Preprocesses the input data, splits it for cross-validation, trains the regression
        algorithm, predicts the output for the test set, and calculates the performance
        metrics for the model.
        """
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

        # Preprocess split data based on reference data for the training data
        preprocessed_split = PreprocessingOnSplitData(
            split=split, x_called=x_called, y_called=y_called
        )
        test_set = preprocessed_split.test_set
        training_set = preprocessed_split.training_set

        # Create models from the preprocessed training data
        models = CrossValidationModel(
            algorithm=self.algorithm,
            n_components=self.n_components,
            training=training_set,
        )

        # Calculate y_hat for the test sets and x_scores
        predictor = CrossValidationPredictor(test_set=test_set, models=models.cv_models)
        y_test = np.concatenate(test_set[1])

        # Calculate the regression metrics for the model
        self.results = MetricsRegression(y_test, predictor.predictor_results)


