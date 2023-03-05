import numpy as np

from misc.handle_data import transform_array_1d_to_2d
from preprocessing.standardisation import Standardisation
from preprocessing.filtering import Filtering
from preprocessing.normalisation import Normalisation
from preprocessing.scaling import Scaling


class Preprocessing1D(Scaling):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(transform_array_1d_to_2d(data))


class Preprocessing2D(Scaling, Normalisation, Filtering, Standardisation):
    pass


Preprocessing = {
    "1D": Preprocessing1D,
    "2D": Preprocessing2D,
}
