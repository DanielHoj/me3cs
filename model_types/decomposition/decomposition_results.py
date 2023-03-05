from typing import Union

import numpy as np

from model_types.decomposition.decomposition_diagnostics import DecompositionDiagnostics
from model_types.decomposition.pca import (
    SVD,
    NIPALS,
    EigenDecomposition,
)


class DecompositionResults(DecompositionDiagnostics):
    def __init__(
        self,
        x: np.ndarray,
        results: Union[SVD, NIPALS, EigenDecomposition],
    ):
        self.__dict__.update(results.__dict__)
        super().__init__(x, results)
        self.scores = results.scores
        self.loadings = results.loadings
        self.explained_variance = results.explained_variance
        self.cumulative_explained_variance = results.explained_variance.cumsum()
