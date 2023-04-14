import numpy as np

from me3cs.framework.helper_classes.options import dict_to_string_with_newline
from me3cs.misc.metrics import residuals, q_residuals, leverage, hotellings_t2
from me3cs.models.regression.pls import SIMPLS, NIPALS


class DiagnosticsPLS:
    def __init__(
            self,
            x: np.ndarray,
            results: [SIMPLS, NIPALS],
    ):
        res = residuals(x, results.x_scores, results.x_loadings)
        self.q_residuals = q_residuals(res)
        self.leverage = leverage(results.x_scores, x.shape[0])
        self.hotelling_t2 = hotellings_t2(results.x_scores)

    def __repr__(self):
        return f"Diagnostics calculated:\n" \
               f"{dict_to_string_with_newline(self.__dict__)}"
