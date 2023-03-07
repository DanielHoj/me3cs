import numpy as np

from me3cs.preprocessing.base import PreprocessingBaseClass, sort_function_order
from me3cs.preprocessing.called import set_called


class Standardisation(PreprocessingBaseClass):
    """
    A class containing standardisation methods for preprocessing.


    Methods:
    -------
    absolute_value():
        Converts the data to absolute values.
    arithmic_operation(func, *args, variable_range=None):
        Performs an arithmetic operation on the data within the specified range.
    log10():
        Calculates the logarithm to the base 10 of the data.
    glog(lambd=1.00e-09, data_0=0):
        Performs the generalized logarithm transformation on the data.
    t2a():
        Calculates the inverse logarithm of the data.

    """
    @sort_function_order
    @set_called
    def absolute_value(self) -> None:
        """
        Converts the data to absolute values.
        """
        data = self.data
        new = np.abs(data)
        self.data = new

    @sort_function_order
    @set_called
    def arithmic_operation(
            self, func, *args, variable_range: [list | tuple] = None
    ) -> None:
        """
        Performs an arithmetic operation on the data within the specified range.

        Parameters:
        ----------
        func : function
            The arithmetic function to be performed on the data.
        *args : arguments
            Additional arguments to be passed to the arithmetic function.
        variable_range : list or tuple, optional
            The range of data on which to perform the arithmetic operation. The default is None (the whole
            length of the data set).

        Raises:
        ------
        TypeError
            If variable_range is not a list or tuple.
        """
        data = self.data

        if variable_range is None:
            variable_range = [0, data.shape[1]]
        if not isinstance(variable_range, (list, tuple)):
            raise TypeError("Please input list or tuple as variable_range")

        new = data
        new[:, variable_range[0]: variable_range[1]] = func(
            data[:, variable_range[0]: variable_range[1]], args
        )

        self.data = new

    @sort_function_order
    @set_called
    def log10(self) -> None:
        """
        Take the logarithm base 10 of the data.
        """
        data = self.data

        data = data.clip(min=0)

        self.data = np.log10(data)

    @sort_function_order
    @set_called
    def glog(self, lambd: float = 1.00e-09, data_0: float = 0) -> None:
        """
        Apply the generalized logarithm to the data.

        Parameters
        ----------
        lambd : float, optional
            Regularisation parameter. Default is 1.00e-09.
        data_0 : float, optional
            Value to shift the data with. Default is 0.
        """
        data = self.data

        new = np.log((data - data_0) + np.sqrt((data - data_0) ** 2 + lambd))
        self.data = new

    @sort_function_order
    @set_called
    def t2a(self) -> None:
        """
        Transform the data to absorbance values.
        """

        data = self.data

        new = np.log10(1 / data)

        self.data = new
