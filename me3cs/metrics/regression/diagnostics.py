from me3cs.misc.metrics import residuals, q_residuals, leverage, hotellings_t2
from me3cs.models.regression.pls import SIMPLS, NIPALS


class DiagnosticsPLS:
    def __init__(
            self,
            results: [SIMPLS, NIPALS],
    ):
        res = residuals(results.x, results.x_scores, results.x_loadings)
        self.q_residuals = q_residuals(res)
        self.leverage = leverage(results.x_scores, results.x.shape[0])
        self.hotelling_t2 = hotellings_t2(results.x_scores)

    def __repr__(self):
        diagnostics = ", ".join(self.__dict__.keys())
        return f"Diagnostics calculated:\n" \
               f"{diagnostics}"
