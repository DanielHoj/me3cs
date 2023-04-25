import warnings

import numpy as np
import typing

from me3cs.framework.data import Data, count_false
from me3cs.missing_data.imputation import imputation_algorithms
from me3cs.missing_data.interpolation import interpolation_algorithms
from me3cs.preprocessing.called import Called, set_called

if typing.TYPE_CHECKING:
    from me3cs.framework.branch import Branch


def check_nan(data: np.ndarray) -> None:
    """
    Check if the input array contains any NaN values and raise a warning if so.

    Parameters
    ----------
    data : np.ndarray
        Input array to be checked for NaN values.

    Warns
    -----
    UserWarning
        If `data` contains NaN values.

    """
    has_nan = np.isnan(data).any()
    if has_nan:
        warnings.warn("Dataset contain missing values. Consider using the missing_values module.")


class MissingData:
    """
    A class for handling missing data in numpy arrays.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to handle missing values.
    branches : list of Branches
        An object to keep track of the data linkage, by default None.

    Attributes
    ----------
    data : numpy.ndarray
        The current data array.

    Raises
    ------
    ValueError
        If the input dimension is not 0 or 1 in the `delete_nan` method.
        If there are no more missing values to delete in the `delete_nan` method.
    TypeError
        If the input algorithm is not a string in the `_check_algorithm_type` method.
        If the input algorithm is not in the specified algorithm type in the `_check_algorithm_type` method.

    Methods
    -------
    interpolation():
        Interpolate missing values using a specified algorithm.
    imputation():
        Impute missing values using a specified algorithm.
    delete_nan():
        Delete rows or columns containing missing values. The default is rows (dim=0).
    reset() -> None:
        Reset the data to the raw data.

    """

    def __init__(self, data: Data,
                 branches: list["Branch", ...]) -> None:
        """
        Initialize the MissingData object.
        """
        self.data_class = data
        check_nan(self.data)
        self._branches = branches
        self.called = Called(list(), list(), list())

    @property
    def data(self):
        return self.data_class.data

    @data.getter
    def data(self):
        return self.data_class.data

    @data.setter
    def data(self, data):
        self.data_class.missing_data.set(data)

    @set_called
    def interpolation(self, algorithm: str = "mean") -> None:
        """
        Interpolate missing values using a specified algorithm.

        Parameters
        ----------
        algorithm : str, optional
            The algorithm to use for interpolation, by default "mean".

        Raises
        ------
        TypeError
            If the input algorithm is not in the specified algorithm type in the `_check_algorithm_type` method.
        """
        self._check_algorithm_type(algorithm, interpolation_algorithms)
        func = interpolation_algorithms.get(algorithm)
        result = func(self.data_class.raw.get())
        self.data = result

    @set_called
    def imputation(self, algorithm: str = "emsvd") -> None:
        """
        imputate missing values using a specified algorithm.

        Parameters
        ----------
        algorithm : str, optional
            The algorithm to use for interpolation, by default "emsvd".

        Raises
        ------
        TypeError
            If the input algorithm is not in the specified algorithm type in the `_check_algorithm_type` method.

        """
        self._check_algorithm_type(algorithm, imputation_algorithms)
        func = imputation_algorithms.get(algorithm)
        result = func(self.data_class.raw.get())
        self.data = result

    @set_called
    def remove_nan(self, dim: int = 0) -> None:
        """
        Delete missing values in the dataset along a specified dimension.

        Parameters
        ----------
        dim : int, default 0
            Dimension along which missing values should be removed.
            0 corresponds to rows, 1 corresponds to columns.

        Raises
        ------
        ValueError
            If `dim` is not 0 or 1, or if there are no missing values along the specified dimension.
        """
        if dim not in [1, 0]:
            raise ValueError(f'Please input 1 or 0. {dim} was input')

        if dim == 1:
            if np.count_nonzero(~np.isnan(self.data).any(axis=0)) == self.data.shape[1]:
                raise ValueError("No more missing values")
            missing_values = count_false(~np.isnan(self.data).any(axis=0))
            [branch.data_class.remove_columns("missing_data", missing_values) for branch in self._branches]

        else:
            if np.count_nonzero(~np.isnan(self.data).any(axis=1)) == self.data.shape[0]:
                raise ValueError("No more missing values")
            missing_values = count_false(~np.isnan(self.data).any(axis=1))
            [branch.data_class.remove_rows("missing_data", missing_values) for branch in self._branches]

    @staticmethod
    def _check_algorithm_type(algorithm_input: str, algorithm_type: [interpolation_algorithms, imputation_algorithms]):
        """
        Check whether the specified algorithm input is valid.

        Parameters
        ----------
        algorithm_input : str
            The name of the algorithm to check.
        algorithm_type : [interpolation_algorithms, imputation_algorithms]
            The types of algorithms to check against.

        Raises
        ------
        TypeError
            If the algorithm input is not a string or if it is not a valid option in the specified algorithm type.

        """
        algorithm_type = list(algorithm_type.keys())
        if not isinstance(algorithm_input, str):
            raise TypeError(f"Please input a string of {algorithm_type}")
        if algorithm_input not in algorithm_type:
            raise TypeError(f"Please choose one of {algorithm_type}. {algorithm_input} was chosen")

    def reset(self) -> None:
        """
        resets the data to the `raw data`.
        """
        [branch.data_class.reset_index("all") for branch in self._branches]
        self.called.reset()

    def call_in_order(self):
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            function(self, *args, **kwargs)

    def __repr__(self):
        return f"Missing data module\n" \
               f"Nr of missing values: {np.sum(np.isnan(self.data))}\n" \
               f"Nr of missing values removed: " \
               f"{np.sum(np.isnan(self.data_class.raw.get())) - np.sum(np.isnan(self.data))}\n" \
               f"{self.called}\n"
