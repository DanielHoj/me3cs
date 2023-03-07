import numpy as np
import pandas as pd

from me3cs.framework.helper_classes.link import Link


class BaseGetter:
    def __init__(self, data_link: Link):
        """
        Constructs a BaseGetter instance with a Link object that holds data.

        Parameters
        ----------
        data_link : Link
            The Link object that holds data.

        Returns
        -------
        BaseGetter
            An instance of the BaseGetter class.
        """
        self._data_link = data_link

    @property
    def data(self):
        """
        Returns the data held in the Link object.

        Returns
        -------
        np.ndarray or pd.Series or pd.DataFrame
            The data held in the Link object.
        """
        return self._data_link.get()

    @data.setter
    def data(self, data):
        """
        Sets the data held in the Link object to the given data.

        Parameters
        ----------
        data : np.ndarray or pd.Series or pd.DataFrame
            The data to be set in the Link object.

        Raises
        ------
        ValueError
            If the input data is not of type np.ndarray, pd.Series or pd.DataFrame.
        """
        if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError('Please input np.ndarray, pd.Series or pd.DataFrame')
        else:
            self._data_link.set(data)

    @data.getter
    def data(self):
        """
        Returns the data held in the Link object.

        Returns
        -------
        np.ndarray or pd.Series or pd.DataFrame
            The data held in the Link object.
        """
        return self._data_link.get()
