import numpy as np

from me3cs.misc.handle_data import handle_zeros_in_scale
from me3cs.misc.preprocessing import preprocessing_scaling
from me3cs.preprocessing.base import PreprocessingBaseClass, sort_function_order
from me3cs.preprocessing.called import set_called


class Normalisation(PreprocessingBaseClass):
    """
    A class containing normalisation methods for preprocessing.


    Methods:
    --------
    snv():
        Perform Standard Normal Variate (SNV) scaling on the spectral data.

    msc(reference: np.ndarray = None):
        Perform Multiplicative Scatter Correction (MSC) on the spectral data.
    """

    @sort_function_order
    @set_called
    def snv(self) -> None:
        """
        Perform Standard Normal Variate (SNV) scaling on the spectral data.
        """
        data = self.data
        constant = -data.mean(axis=1)
        scale = handle_zeros_in_scale(data.std(axis=1))

        new = preprocessing_scaling(data.T, constant, scale)
        self.data = new.T

    @sort_function_order
    @set_called
    def msc(self, reference: np.ndarray = None) -> None:
        """
        Perform Multiplicative Scatter Correction (MSC) on the spectral data.

        Parameters:
        -----------
        reference : np.ndarray, optional
            A numpy array containing the reference spectrum. If not provided, the mean
            of the spectral data will be used as reference.
        """
        data = self.data

        if reference is None:
            # Set reference data
            ref = data.mean(axis=0)

        else:
            ref = reference

        fit = np.polynomial.polynomial.polyfit(ref, data.T, deg=1)
        new = (data.T - fit[0]) / handle_zeros_in_scale(fit[1])

        self.data = new.T
