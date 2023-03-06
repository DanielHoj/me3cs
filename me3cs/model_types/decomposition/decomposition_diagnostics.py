import numpy as np

from me3cs.misc.metrics import confidence_limit_matrix, residuals, q_residuals
from me3cs.model_types.decomposition.pca import (
    NIPALS,
    EigenDecomposition,
    SVD,
)


class DecompositionDiagnostics:
    def __init__(
        self,
        x: np.ndarray,
        results: [SVD, NIPALS, EigenDecomposition],
    ):
        self.residuals = residuals(x, results.scores, results.loadings)
        self.q_residuals = q_residuals(self.residuals)
        self.hotelling_confidence_limit = confidence_limit_matrix(self.hotelling_t2)
        self.q_residual_confidence_limit = confidence_limit_matrix(self.q_residuals)
