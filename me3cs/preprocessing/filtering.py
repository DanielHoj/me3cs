import numpy as np
from scipy.ndimage import convolve1d

from me3cs.misc.preprocessing import savgol_coefficients
from me3cs.preprocessing.base import PreprocessingBaseClass, sort_function_order
from me3cs.preprocessing.called import set_called


class Filtering(PreprocessingBaseClass):
    """
    Class for preprocessing data, with filtering algorithms.

    Methods
    -------
    savitzky_golay(width=15, polyorder=2, deriv=1, delta=1)
        Filter data using the Savitzky-Golay algorithm.
    baseline(polyorder=1, value_range=None, fit_type='data')
        Perform baseline correction on data.

    """
    @sort_function_order
    @set_called
    def savitzky_golay(
        self, width: int = 15, polyorder: int = 2, deriv: int = 1, delta: int = 1
    ) -> None:
        """
        Filter data using the Savitzky-Golay algorithm.

        Parameters
        ----------
        width : int, optional
            The number of points to use for filtering, by default 15.
        polyorder : int, optional
            The order of the polynomial to use for filtering, by default 2.
        deriv : int, optional
            The derivative order to use for filtering, by default 1.
        delta : int, optional
            The spacing between points to use for filtering, by default 1.

        Raises
        ------
        ValueError
            If width is not an odd number greater or equal to 3 or if deriv is greater than polyorder.

        """
        if width < 3 or width % 2 == 0:
            raise ValueError("width needs to be odd and greater or equal to 3")

        if polyorder < deriv:
            raise ValueError("deriv needs to be smaller or equal to order")

        data = self.data
        coeffs = savgol_coefficients(width, polyorder, deriv, delta)

        new = convolve1d(data, coeffs)

        self.data = new

    @sort_function_order
    @set_called
    def baseline(
        self, polyorder: int = 1, value_range: tuple = None, fit_type: str = "data"
    ) -> None:
        """
        Perform baseline correction on data.

        Parameters
        ----------
        polyorder : int, optional
            The order of the polynomial to use for fitting, by default 1.
        value_range : tuple of int, optional
            The range of values to fit the polynomial to, by default None (uses the entire data range).
        fit_type : str, optional
            The type of data to use for fitting. "data" uses the original data, "mean" uses the mean of the data,
            by default "data".

        Raises
        ------
        ValueError
            If value_range is not a tuple of length 2 or if fit_type is not "data" or "mean".
        TypeError
            If value_range is not a list or tuple.

        """
        data = self.data

        if value_range is None:
            value_range = (0, data.shape[1])

        if not isinstance(value_range, (list, tuple)):
            raise TypeError("Please input list or tuple")
        if not len(value_range) == 2:
            raise ValueError(
                f"Please input list or tuple of length 2. Length of value_range is {value_range}."
            )

        if fit_type not in ["data", "mean"]:
            raise ValueError(
                f'Please input "data" or "mean" as fit_type. {fit_type} was input.'
            )

        if fit_type == "data":
            poly_x = data
        else:
            poly_x = np.tile(data.mean(axis=0), data.shape[0]).reshape(data.shape)

        min_range, max_range = value_range
        poly_x = poly_x[:, min_range:max_range]

        a = np.linspace(min_range, 1, max_range)
        a_full = np.linspace(0, 1, data.shape[1])

        coef = np.polynomial.polynomial.polyfit(a, poly_x.T, deg=polyorder)

        baseline = np.polynomial.polynomial.polyval(a_full, coef)
        new = data - baseline

        self.data = new
