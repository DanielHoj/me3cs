from me3cs.framework.helper_classes.link import LinkedBranches
from me3cs.misc.metrics import leverage, residuals, q_residuals
from me3cs.model_types.regression.pls import SIMPLS, NIPALS


class OutlierDetection:
    def __init__(self, results: [SIMPLS, NIPALS], linked_branches: LinkedBranches):
        self._linked_branches = linked_branches

        self.leverage = leverage(results.x_scores, results.x.shape[0])
        self.residuals = residuals(results.x, results.x_scores, results.x_loadings)
        self.q_residuals = q_residuals(self.residuals)

    def remove_outlier(self, name: str):
        all_atributes = [attr for attr in dir(self) if
                         not callable(getattr(self, attr))
                         and not attr.startswith("__")
                         and not attr.startswith("_")]

        if not hasattr(self, name):
            raise AttributeError(f"Input needs to be one of {all_atributes}. \n"
                                 f"{name} was input")


class Results:
    def __init__(self, results: [SIMPLS, NIPALS], linked_branches: LinkedBranches):
        self.calibration = None
        self.cross_validation = None
        self.outlier_detection = OutlierDetection(results, linked_branches)
