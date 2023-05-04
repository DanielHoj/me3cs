import numpy as np
import pandas as pd

from me3cs.framework.branch import Branch
from me3cs.framework.data import Data, Index
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.log import Log
from me3cs.framework.outlier_detection import OutlierDetection
from me3cs.framework.results import Results
from me3cs.framework.variable_selection import VariableSelection
from me3cs.misc.handle_data import transform_array_1d_to_2d


class BaseModel:
    """
    A base class for building models in the me3cs module. RegressionModel, DecompositionModel and CalibrationModel is
    build on this base class.

    Parameters
    ----------
    x : np.ndarray, pd.Series, or pd.DataFrame
        The input data array for the model.
    y : np.ndarray, pd.Series, or pd.DataFrame, optional
        The reference data array for the model, by default None.

    Attributes
    ----------
    branches : list
        A list of Branch objects for each data array.
    x : Branch
        The Branch object for the input data array x.
    y : Branch, optional
        The Branch object for the target data array y, by default None.
    single_branch : bool
        Whether the model has a single branch (x only) or not (x and y).
    results : Results
        The Results object for storing the model results.
    options : Options
        The Options object for storing the model configuration options.
    log : Log
        The Log object for logging events during model operations.
    outlier_detection : OutlierDetection
        The OutlierDetection object for detecting outliers in the data.
    """
    def __init__(
            self,
            x: [np.ndarray | pd.Series | pd.DataFrame],
            y: [np.ndarray | pd.Series | pd.DataFrame] = None,
    ) -> None:
        x_data = Data(x, Index(x.shape[0]), Index(x.shape[1]))

        self.branches = []
        self.x = Branch(x_data, self.branches)
        self.branches.append(self.x)
        self.single_branch = True
        if y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"x and y need to have the same number of rows. "
                                 f"\nx rows {x.shape[0]}"
                                 f"\ny rows {y.shape[0]}")

            y = transform_array_1d_to_2d(y)
            y_data = Data(y, Index(y.shape[0]), Index(y.shape[1]))

            self.y = Branch(y_data, self.branches)
            self.branches.append(self.y)
            self.single_branch = False

        self.results = Results()
        self.options = Options()
        self.log = Log(self, self.results, self.options)
        self.outlier_detection = OutlierDetection(self)
        self.variable_selection = VariableSelection(self)

    def reset(self):
        """
        Reset the model by clearing the outlier detection, last model called,
        and the preprocessing and missing data information for each branch.
        """
        self.outlier_detection.reset()
        self.log.log_object.last_model_called = None
        [branch.preprocessing.reset() for branch in self.branches]
        [branch.missing_data.reset() for branch in self.branches]

    def __repr__(self) -> str:
        """
        Return a string representation of the BaseModel object.

        Returns
        -------
        str
            A string representation of the BaseModel object.
        """
        return f"me3cs Model"
