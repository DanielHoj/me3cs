import numpy as np

from me3cs.misc.handle_data import transform_array_1d_to_2d
from me3cs.preprocessing.standardisation import Standardisation
from me3cs.preprocessing.filtering import Filtering
from me3cs.preprocessing.normalisation import Normalisation
from me3cs.preprocessing.scaling import Scaling


class Preprocessing1D(Scaling):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(transform_array_1d_to_2d(data))


class Preprocessing2D(Scaling, Normalisation, Filtering, Standardisation):
    pass


Preprocessing = {
    "1D": Preprocessing1D,
    "2D": Preprocessing2D,
}
