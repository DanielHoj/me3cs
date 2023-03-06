from typing import Union

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


class CrossValidationRegression:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        called_preprocessing: Union[tuple[Called, Called], Called],
        algorithm: any,
        n_components: int,
        cv_type: str,
        cv_metrics,
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
        partly_preprocessed_x = PreSplitPreprocessing(
            data=self.x, called=x_called
        ).data

        split = CrossValidationSplit(
            x=partly_preprocessed_x,
            y=self.y,
            percentage_left_out=self.percentage_left_out,
            _cv_type=self.cv_type,
        )

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


class CrossValidationDecomposition:
    def __init__(
        self,
        x: np.ndarray,
        called_preprocessing: Called,
        algorithm: any,
        n_components: int,
        cv_type: str,
        cv_metrics,
        percentage_left_out: float = 0.1,
    ) -> None:
        self.x = x
        self.called_preprocessing = called_preprocessing
        self.percentage_left_out = percentage_left_out
        self.algorithm = algorithm
        self.n_components = n_components
        self.cv_type = cv_type
        self.cv_metrics = cv_metrics
        self.results = None
        self.fit()

    def fit(self) -> None:
        x_called = self.called_preprocessing
        partly_preprocessed_x = PreSplitPreprocessing(
            data=self.x, called=x_called
        ).data

        split = CrossValidationSplit(
            x=partly_preprocessed_x,
            percentage_left_out=self.percentage_left_out,
            _cv_type=self.cv_type,
        )

        preprocessed_split = PreprocessingOnSplitData(split=split, x_called=x_called)

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


CrossValidation = {
    "regression": CrossValidationRegression,
    "decomposition": CrossValidationDecomposition,
}
