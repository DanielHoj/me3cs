import numpy as np
import pandas as pd

from framework.helper_classes.link import Link


class BaseGetter:
    def __init__(self, data_link: Link):
        self._data_link = data_link

    @property
    def data(self):
        return self._data_link.get()

    @data.setter
    def data(self, data):
        if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError('Please input np.ndarray, pd.Series or pd.DataFrame')
        else:
            self._data_link.set(data)

    @data.getter
    def data(self):
        return self._data_link.get()
