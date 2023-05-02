from typing import TYPE_CHECKING

import numpy as np

from me3cs.cross_validation.cross_validation import CrossValidationRegression
from me3cs.framework.base_model import BaseModel
from me3cs.framework.outlier_detection import choose_optimal_component
from me3cs.metrics.regression.diagnostics import DiagnosticsPLS
from me3cs.metrics.regression.metrics import MetricsRegression
from me3cs.metrics.regression.results import RegressionResults
from me3cs.models.regression import MLR, PCR, PLS

if TYPE_CHECKING:
    from me3cs.models.regression import TYPING_ALGORITHM_REGRESSION
    from me3cs.metrics.regression import TYPING_RESULTS_REGRESSION


class RegressionModel(BaseModel):
    """
    Class for regression model analysis, including PLS, PCR, MLR.
    """

    def pls(
            self,
            algorithm: str = "SIMPLS",
    ) -> None:
        """
        Perform partial least squares (PLS) regression analysis with the specified algorithm.

        Parameters
        ----------
        algorithm : str, optional
            PLS algorithm to use, default is "SIMPLS". Implemented algorithms are SIMPLS and NIPALS
        """
        if algorithm not in list(PLS.keys()):
            raise ValueError(
                f"Please input {list(PLS.keys())} as algorithm. {algorithm} was input"
            )
        # Get algorithm
        algorithm = PLS[f"{algorithm}"]
        reg_results = RegressionResults["PLS"]

        self.__regresion_pileline__(algorithm=algorithm, reg_results=reg_results)

    def pcr(self):
        """
        Perform principal component regression (PCR) analysis.
        """
        # Get algorithm
        algorithm = PCR
        reg_results = RegressionResults["PCR"]

        self.__regresion_pileline__(algorithm=algorithm, reg_results=reg_results)

    def mlr(self):
        """
        Perform multiple linear regression (MLR) analysis.
        """
        # Get algorithm
        algorithm = MLR
        reg_results = RegressionResults["MLR"]

        self.__regresion_pileline__(algorithm=algorithm, reg_results=reg_results)

    def cls(self):
        # TODO: implement cls algorithm
        pass

    def svm(self):
        # TODO: implement svm algorithm
        pass

    def __regresion_pileline__(self, algorithm: "TYPING_ALGORITHM_REGRESSION",
                               reg_results: "TYPING_RESULTS_REGRESSION") -> None:
        """
        Perform regression analysis using the provided algorithm and store the results in the RegressionModel instance.

        Parameters
        ----------
        algorithm : TYPING_ALGORITHM_REGRESSION
            The regression algorithm to use.
        reg_results : REGRESSION_RESULTS_TYPES
            The results container for the specific algorithm.
        """
        # Get raw data
        x = self.x.data_class.get_raw_data()
        y = self.y.data_class.get_raw_data()

        if np.isnan(x).any():
            raise ValueError("x contains missing values. Use the missing_data module to adress the problem")

        if np.isnan(y).any():
            raise ValueError("y contains missing values. Use the missing_data module to adress the problem")

        # mean center if not mean centered
        if not self.x.preprocessing.data_is_centered:
            if self.options.mean_center:
                self.x.preprocessing.mean_center()

        if not self.y.preprocessing.data_is_centered:
            if self.options.mean_center:
                self.y.preprocessing.mean_center()

        # Get called preprocessing
        x_preprocessing = self.x.preprocessing.called
        y_preprocessing = self.y.preprocessing.called
        called_preprocessing = (x_preprocessing, y_preprocessing)

        cv = CrossValidationRegression(  # Create entries with the cross-validation module
            x=x,
            y=y,
            called_preprocessing=called_preprocessing,
            algorithm=algorithm,
            n_components=self.options.n_components,
            cv_type=self.options.cross_validation,
            cv_metrics=MetricsRegression,
            percentage_left_out=self.options.percentage_left_out,
        )

        # Get preprocessed data
        x_prep = self.x.data
        y_prep = self.y.data

        model = algorithm(  # Create calibration model
            x=x_prep, y=y_prep, n_components=self.options.n_components
        )

        calibration_results = reg_results(x_prep, y_prep, model)
        diagnostics = DiagnosticsPLS(x_prep, calibration_results)
        n_components = choose_optimal_component(calibration_results.rmse, cv.results.rmse)

        self.log.log_object.last_model_called = "PLS"

        # Set calibration and cross-validation results
        setattr(self.results, "cross_validation", cv.results)
        setattr(self.results, "calibration", calibration_results)
        setattr(self.results, "diagnostics", diagnostics)
        setattr(self.results, "optimal_number_component", n_components)

        self.log.make_entry()
