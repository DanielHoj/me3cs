import numpy as np

from me3cs.misc.handle_data import transform_array_1d_to_2d
from me3cs.preprocessing.standardisation import Standardisation
from me3cs.preprocessing.filtering import Filtering
from me3cs.preprocessing.normalisation import Normalisation
from me3cs.preprocessing.scaling import Scaling


class Preprocessing1D(Scaling):
    """
    Class for preprocessing 1-dimensional data.

    Inherits from `Scaling` class in `me3cs.preprocessing.scaling`.
    """
    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize a `Preprocessing1D` object with the given 1-dimensional data.
        The data is automatically transformed from (n,) into (n,1)

        Parameters
        ----------
        data : np.ndarray
            The 1-dimensional data to be preprocessed.
        """
        super().__init__(transform_array_1d_to_2d(data))


class Preprocessing2D(Scaling, Normalisation, Filtering, Standardisation):
    """
    Class for preprocessing 2-dimensional data.

    Inherits from `Scaling`, `Normalisation`, `Filtering`, and `Standardisation` classes
    in `me3cs.preprocessing`.
    """
    pass


Preprocessing = {
    "1D": Preprocessing1D,
    "2D": Preprocessing2D,
}
"""
A dictionary that maps string keys to corresponding preprocessing classes.

Keys:
    - '1D': `Preprocessing1D` class
    - '2D': `Preprocessing2D` class
"""
