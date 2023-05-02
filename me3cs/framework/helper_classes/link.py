from dataclasses import dataclass

import numpy as np


@dataclass
class Link:
    """
    A dataclass for storing a single piece of data. Mainly used as a link between the modules in the branch class.

    Parameters
    ----------
    data : np.ndarray, optional
        The data to be stored.
    """
    data: [np.ndarray, None]

    def get(self):
        """
        Return the data stored in the Link object.

        Returns
        -------
        np.ndarray or None
            The data stored in the Link object.
        """
        return self.data

    def set(self, data):
        """
        Set the data stored in the Link object to the input data.

        Parameters
        ----------
        data : np.ndarray, optional
            The data to be stored in the Link object, by default None.
        """
        self.data = data

