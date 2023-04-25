from dataclasses import dataclass

import numpy as np


@dataclass
class Link:
    """
    A dataclass for storing a single piece of data. Mainly used as a link between the modules in the branch class.

    Parameters
    ----------
    data : {np.ndarray, None}
        The data to be stored.
    """
    data: [np.ndarray, None]

    def get(self):
        """
        Returns the data stored in the Link object.

        Returns
        -------
        {np.ndarray, None}
            The data stored in the Link object.
        """
        return self.data

    def set(self, data):
        """
        Sets the data stored in the Link object to the input data.

        Parameters
        ----------
        data : {np.ndarray, None}
            The data to be stored in the Link object.
        """
        self.data = data

