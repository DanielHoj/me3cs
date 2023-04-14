import numpy as np
import pandas as pd

from me3cs.framework.log import Log
from me3cs.framework.outlier_detection import OutlierDetection
from me3cs.framework.results import Results

from me3cs.framework.helper_classes.link import LinkedBranches
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.branch import Branch


class BaseModel:
    def __init__(
            self,
            x: [np.ndarray | pd.Series | pd.DataFrame],
            y: [np.ndarray | pd.Series | pd.DataFrame] = None,
    ) -> None:
        self._linked_branches = LinkedBranches([])
        self.x = Branch(x, self._linked_branches)
        self._linked_branches.add_branch(self.x)
        self._single_branch = True
        if y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"x and y need to have the same number of rows. "
                                 f"\nx rows {x.shape[0]}"
                                 f"\ny rows {y.shape[0]}")
            self.y = Branch(y, self._linked_branches)
            self._linked_branches.add_branch(self.y)
            self._single_branch = False

        self.results = Results(self._linked_branches)
        self.options = Options()
        self.log = Log(self, self.results, self.options)
        self.outlier_detection = OutlierDetection(self)

    def reset(self):
        self.outlier_detection.reset()
        self.log.log_object.last_model_called = None
        [branch.preprocessing.reset() for branch in self._linked_branches.branches]
        [branch.missing_data.reset() for branch in self._linked_branches.branches]

    def __repr__(self) -> str:
        return f"me3cs Model"
