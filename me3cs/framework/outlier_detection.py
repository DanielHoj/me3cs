import typing

import numpy as np
import scipy.interpolate as interpolate
from scipy.signal import argrelextrema

if typing.TYPE_CHECKING:
    from me3cs.framework.base_model import BaseModel


class OutlierDetection:
    """
    Class for detecting and removing outliers from the model.

    Parameters
    ----------
    model : BaseModel
        The model for which outlier detection should be performed.
    """
    def __init__(self, model: "BaseModel"):
        self._branches = model.branches
        self._result = model.results
        self._model = model

    def _remove_outlier_from(self, number_of_outliers_to_remove: int, diagnostic_name: str):
        """
        Removes the specified number of outliers from the data using the given diagnostic name.
        """
        if self._result.diagnostics is None:
            raise ReferenceError("diagnostics are not calculated")
        elif not hasattr(self._result.diagnostics, diagnostic_name):
            raise ValueError(f"{diagnostic_name} is not calculated")

        if not isinstance(self._result.optimal_number_component, (int, np.int64)):
            raise ValueError("optimal number of components must be chosen")

        opt_component = self._result.optimal_number_component - 1

        diagnostic = getattr(self._result.diagnostics, diagnostic_name)
        diagnostic_optimal = diagnostic[:, opt_component]
        diagnostic_optimal = diagnostic_optimal.argsort()
        outliers_to_remove = diagnostic_optimal[-number_of_outliers_to_remove:]
        self.remove_outliers(tuple(outliers_to_remove))

    def remove_outliers(self, outlier_index: [tuple[..., int], int]) -> None:
        """
        Removes the outliers specified by the given indices.

        Parameters
        ----------
        outlier_index : tuple[int] or int
            A tuple or single integer representing the indices of the outliers to be removed.
        """
        if not isinstance(outlier_index, (tuple, int)):
            raise TypeError("Input needs to be an int or a tuple of ints")
        if isinstance(outlier_index, tuple) and not all(isinstance(x, (int, np.int64)) for x in outlier_index):
            raise TypeError("Input needs to be an int or a tuple of ints")

        [branch.data_class.remove_rows("outlier_detection", outlier_index) for branch in self._branches]
        [branch.preprocessing.call_in_order() for branch in self._branches]
        call_model(self)

    def remove_outlier_from_q_residuals(self, number_of_outliers_to_remove: int = 1):
        """
        Removes the specified number of outliers from the data based on the Q residuals for the optimal component.

        Parameters
        ----------
        number_of_outliers_to_remove : int, optional
            The number of outliers to remove, by default 1.
        """
        self._remove_outlier_from(number_of_outliers_to_remove, "q_residuals")

    def remove_outlier_from_hotellings_t2(self, number_of_outliers_to_remove: int = 1):
        """
        Removes the specified number of outliers from the data based on the Hotelling's T2 statistic for the optimal
        component.

        Parameters
        ----------
        number_of_outliers_to_remove : int, optional
            The number of outliers to remove, by default 1.
        """
        self._remove_outlier_from(number_of_outliers_to_remove, "hotelling_t2")

    def remove_outlier_from_leverage(self, number_of_outliers_to_remove: int = 1):
        """
        Removes the specified number of outliers from the data based on the leverage statistic for the optimal
        component.

        Parameters
        ----------
        number_of_outliers_to_remove : int, optional
            The number of outliers to remove, by default 1.
        """
        self._remove_outlier_from(number_of_outliers_to_remove, "leverage")

    def reset(self):
        """
        Resets the outlier detection by removing any previously removed outliers.
        """
        [branch.data_class.reset_index("outlier_detection", dimension="rows") for branch in self._branches]
        [branch.preprocessing.call_in_order() for branch in self._branches]
        call_model(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the OutlierDetection object.

        Returns
        -------
        str
            A string representation of the Branch object.
        """
        return f"Data shape: {self._model.x.data.shape}\n"


def call_model(self):
    if self._model.log.log_object.last_model_called:
        mdl_type = self._model.log.log_object.last_model_called.lower()
        model = getattr(self._model, mdl_type)
        model()


class FindKnee:
    """
    Finds and returns the knee point in the curve.
    """
    def __init__(self, rmse: np.ndarray):
        self.y = rmse
        self.x = np.arange(rmse.shape[0])

        # Step 1: fit a smooth line
        uspline = interpolate.interp1d(self.x, self.y)
        self.Ds_y = uspline(self.x)

        self._y_normalised = normalise(self.y)
        self._x_normalised = normalise(self.x)

        self._y_normalised = self._y_normalised.max() - self._y_normalised

        self._y_difference = self._y_normalised - self._x_normalised
        self._x_difference = self._x_normalised.copy()

        # local maxima
        self.maxima_indices = argrelextrema(self._y_difference, np.greater_equal)[0]
        self.x_difference_maxima = self._x_difference[self.maxima_indices]
        self.y_difference_maxima = self._y_difference[self.maxima_indices]

        # local minima
        self.minima_indices = argrelextrema(self._y_difference, np.less_equal)[0]
        self.x_difference_minima = self._x_difference[self.minima_indices]
        self.y_difference_minima = self._y_difference[self.minima_indices]

        self.Tmx = self.y_difference_maxima - (
            np.abs(np.diff(self._x_normalised).mean())
        )

        self.knee = self.find_knee()

    def find_knee(self) -> int:
        # If no
        if not self.maxima_indices:
            return 0

        # placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        threshold = 0
        threshold_index = 0

        # traverse the difference curve
        for i, x in enumerate(self._x_difference):
            # skip points on the curve before the first local maxima
            if i < self.maxima_indices[0]:
                continue

            j = i + 1

            # reached the end of the curve
            if x == 1.0:
                return 0

            # if we're at a local max, increment the maxima threshold index and continue
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1

            # values in difference curve are at or after a local minimum
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1

            if self._y_difference[j] < threshold:
                knee = self.x[threshold_index]

                return knee


def choose_optimal_component(rmsec: np.ndarray, rmsecv: np.ndarray) -> int:
    """
    Chooses the optimal number of components based on RMSEC and RMSECV values.

    Parameters
    ----------
    rmsec : numpy.ndarray
        The root mean squared error of calibration array.
    rmsecv : numpy.ndarray
        The root mean squared error of cross-validation array.

    Returns
    -------
    int
        The optimal number of components.
    """
    # Find knee of rmsec
    rmsec_knee = FindKnee(rmsec).knee

    # Calculate threshold
    threshold = np.mean(np.abs(np.diff(rmsecv)))

    rmsecv_diff = np.abs(np.diff(rmsecv))
    rmsecv_diff = rmsecv_diff[rmsec_knee:]

    for i, c in enumerate(rmsecv_diff):
        if not c > threshold:
            return i + rmsec_knee + 1


def normalise(x: np.ndarray) -> np.ndarray:
    """
    Normalises an input array by scaling its values to the range [0, 1].

    Parameters
    ----------
    x : numpy.ndarray
        The input array to be normalised.

    Returns
    -------
    numpy.ndarray
        The normalised input array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))
