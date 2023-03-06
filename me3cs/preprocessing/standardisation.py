import numpy as np

from me3cs.preprocessing.base import PreprocessingBaseClass, sort_function_order
from me3cs.preprocessing.called import set_called


class Standardisation(PreprocessingBaseClass):
    @sort_function_order
    @set_called
    def absolute_value(self) -> None:
        data = self.data
        new = np.abs(data)
        self.data = new

    @sort_function_order
    @set_called
    def arithmic_operation(
            self, func, *args, variable_range: [list | tuple] = None
    ) -> None:
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
        data = self.data

        data = data.clip(min=0)

        self.data = np.log10(data)

    @sort_function_order
    @set_called
    def glog(self, lambd: float = 1.00e-09, data_0: float = 0) -> None:
        data = self.data

        new = np.log((data - data_0) + np.sqrt((data - data_0) ** 2 + lambd))
        self.data = new

    @sort_function_order
    @set_called
    def t2a(self) -> None:

        data = self.data

        new = np.log10(1 / data)

        self.data = new
