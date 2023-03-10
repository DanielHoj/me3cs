import numpy as np

from me3cs.models.decomposition.decomposition_diagnostics import DecompositionDiagnostics
from me3cs.models.decomposition.pca import (
    SVD,
    NIPALS,
    EigenDecomposition,
)


class DecompositionResults(DecompositionDiagnostics):
    def __init__(
        self,
        x: np.ndarray,
        results: [SVD, NIPALS, EigenDecomposition],
    ):
        self.__dict__.update(results.__dict__)
        super().__init__(x, results)
        self.scores = results.scores
        self.loadings = results.loadings
        self.explained_variance = results.explained_variance
        self.cumulative_explained_variance = results.explained_variance.cumsum()
