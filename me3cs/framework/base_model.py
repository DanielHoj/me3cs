import numpy as np
import pandas as pd

from me3cs.framework.helper_classes.link import LinkedBranches
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.branch import Branch


class Results:
    pass


class BaseModel:
    def __init__(
            self,
            x: [np.ndarray | pd.Series | pd.DataFrame],
            y: [np.ndarray | pd.Series | pd.DataFrame] = None,
    ) -> None:
        self._linked_branches = LinkedBranches([])
        self.x = Branch(x, self._linked_branches)
        self._linked_branches.add_branch(self.x)

        if y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"x and y need to have the same number of rows. "
                                 f"\nx rows {x.shape[0]}"
                                 f"\ny rows {y.shape[0]}")
            self.y = Branch(y, self._linked_branches)
            self._linked_branches.add_branch(self.y)

        self.results = Results()
        self.options = Options()
