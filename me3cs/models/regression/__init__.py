from typing import Union

from me3cs.models.regression.pls import PLS
from me3cs.models.regression.mlr import MLR
from me3cs.models.regression.pcr import PCR
from me3cs.models.regression.pls import PLS

TYPING_ALGORITHM_REGRESSION = [MLR, PCR]
TYPING_ALGORITHM_REGRESSION.extend(list(PLS.values()))
TYPING_ALGORITHM_REGRESSION = Union[tuple(TYPING_ALGORITHM_REGRESSION)]
