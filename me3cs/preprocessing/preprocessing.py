import numpy as np

from me3cs.preprocessing.filtering import Filtering
from me3cs.preprocessing.normalisation import Normalisation
from me3cs.preprocessing.scaling import Scaling
from me3cs.preprocessing.standardisation import Standardisation


class Preprocessing2D(Scaling, Normalisation, Filtering, Standardisation):
    """
    Class for preprocessing 2-dimensional data.

    Inherits from `Scaling`, `Normalisation`, `Filtering`, and `Standardisation` classes
    in `me3cs.preprocessing`.
    """
    pass


Preprocessing = {
    "1D": Scaling,
    "2D": Preprocessing2D,
}
"""
A dictionary that maps string keys to corresponding preprocessing classes.

Keys:
    - '1D': `Preprocessing1D` class
    - '2D': `Preprocessing2D` class
"""


def get_preprocessing_from_dimension(data: np.ndarray) -> any:
    if data.ndim == 1 or data.shape[1] == 1:
        return Preprocessing["1D"]
    else:
        return Preprocessing["2D"]
