from cross_validation.cross_validation import CrossValidation
from framework.base_model import BaseModel
from metrics.regression.metrics import MetricsRegression
from metrics.regression.results import RegressionResults
from model_types.regression.mlr import MLR
from model_types.regression.pcr import PCR
from model_types.regression.pls import PLS


class RegressionModel(BaseModel):
    def pls(
            self,
            algorithm: str = "SIMPLS",
    ) -> None:
        if algorithm not in list(PLS.keys()):
            raise ValueError(
                f"Please input {list(PLS.keys())} as algorithm. {algorithm} was input"
            )
        # Get algorithm
        algorithm = PLS[f"{algorithm}"]
        reg_results = RegressionResults["PLS"]

        self.__regresion_pileline__(algorithm=algorithm, reg_results=reg_results)

    def pcr(self):
        # Get algorithm
        algorithm = PCR
        reg_results = RegressionResults["PCR"]

        self.__regresion_pileline__(algorithm=algorithm, reg_results=reg_results)

    def mlr(self):

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

    def __regresion_pileline__(self, algorithm: any, reg_results: any):
        # Get raw data
        x = self.x._missing_data_link.get()
        y = self.y._missing_data_link.get()

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

        cv_regression = CrossValidation["regression"]

        cv = cv_regression(  # Create models with the cross-validation module
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

        # Set calibration and cross-validation results
        setattr(self.results, "cross_validation", cv.results)
        setattr(self.results, "calibration", calibration_results)
