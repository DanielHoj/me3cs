import numpy as np

from misc.handle_data import handle_zeros_in_scale
from misc.preprocessing import preprocessing_scaling
from preprocessing.base import PreprocessingBaseClass, sort_function_order
from preprocessing.called import set_called


class Normalisation(PreprocessingBaseClass):
    @sort_function_order
    @set_called
    def snv(self) -> None:
        data = self.data
        constant = -data.mean(axis=1)
        scale = handle_zeros_in_scale(data.std(axis=1))

        new = preprocessing_scaling(data.T, constant, scale)
        self.data = new.T

    @sort_function_order
    @set_called
    def msc(self, reference: np.ndarray = None) -> None:

        data = self.data

        if reference is None:
            # Calculate mean
            ref = data.mean(axis=0)
        else:
            ref = reference

        fit = np.polynomial.polynomial.polyfit(ref, data.T, deg=1)
        new = (data.T - fit[0]) / handle_zeros_in_scale(fit[1])

        self.data = new.T
