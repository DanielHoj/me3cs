import numpy as np
import pandas as pd

from me3cs.framework.branch import Branch
from me3cs.framework.data import Data, Index
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.log import Log
from me3cs.framework.outlier_detection import OutlierDetection
from me3cs.framework.results import Results
from me3cs.misc.handle_data import transform_array_1d_to_2d


class BaseModel:
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

    def reset(self):
        self.outlier_detection.reset()
        self.log.log_object.last_model_called = None
        [branch.preprocessing.reset() for branch in self.branches]
        [branch.missing_data.reset() for branch in self.branches]

    def __repr__(self) -> str:
        return f"me3cs Model"
