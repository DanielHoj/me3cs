from me3cs.misc.metrics import residuals, q_residuals, leverage
from me3cs.models.regression.pls import SIMPLS, NIPALS


class DiagnosticsPLS:
    def __init__(
            self,
            results: [SIMPLS, NIPALS],
    ):
        res = residuals(results.x, results.x_scores, results.x_loadings)
        self.q_residuals = q_residuals(res)
        self.leverage = leverage(results.x_scores, results.x.shape[0])
