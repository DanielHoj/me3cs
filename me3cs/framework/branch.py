from __future__ import annotations

from typing import TYPE_CHECKING

from me3cs.framework.data import Data
from me3cs.missing_data.missing_data import MissingData
from me3cs.preprocessing.preprocessing import get_preprocessing_from_dimension

if TYPE_CHECKING:
    import numpy as np


class Branch:
    """
    A class to represent a data branch in the me3cs module. It is possible to do preprocessing and missing data
    operations in the branch.

    Parameters
    ----------
    data : Data
        The Data object containing the data for the branch.
    branches : list
        A list of Branch objects for each data array.

    Attributes
    ----------
    data_class : Data
        The Data object containing the data for the branch.
    _branches : list
        A list of Branch objects for each data array.
    preprocessing : Preprocessing
        The Preprocessing object for the branch.
    missing_data : MissingData
        The MissingData object for the branch.
    """

    def __init__(self, data: Data, branches: list) -> None:
        preprocessing_type = get_preprocessing_from_dimension(data.data)

        self.data_class = data
        self._branches = branches

        self.preprocessing = preprocessing_type(data)
        self.missing_data = MissingData(data, self._branches)

    @property
    def data(self) -> np.ndarray:
        """
        Get the data array of the branch.

        Returns
        -------
        np.ndarray
            The data array of the branch.
        """
        return self.data_class.data

    @data.getter
    def data(self) -> np.ndarray:
        """
        Get the data array of the branch.

        Returns
        -------
        np.ndarray
            The data array of the branch.
        """
        return self.data_class.data

    @data.setter
    def data(self, data) -> None:
        """
        Set the data array of the branch.

        Parameters
        ----------
        data : np.ndarray, pd.Series, or pd.DataFrame
            The data array to be set for the branch.
        """
        self.data_class.preprocessing_data.set(data)

    def __repr__(self) -> str:
        """
        Return a string representation of the Branch object.

        Returns
        -------
        str
            A string representation of the Branch object.
        """
        return f"Data shape: {self.data.shape}\n" \
               f"Preprocessing - {self.preprocessing.called}\n" \
               f""

    def __getitem__(self, key: [int, slice]) -> Branch:
        """
        Get a new Branch object with a subset of the data array using the given key.

        Parameters
        ----------
        key : int or slice
            The index or slice to subset the data array.

        Returns
        -------
        Branch
            A new Branch object with the subset of the data array.
        """
        return Branch(self.data[key], self._branches)

    def __array__(self) -> np.ndarray:
        """
        Return the data array of the branch.

        Returns
        -------
        np.ndarray
            The data array of the branch.
        """
        return self.data

    def __len__(self) -> int:
        """
        Get the length of the data array of the branch.

        Returns
        -------
        int
            The length of the data array of the branch.
        """
        return len(self.data)
