import numpy as np

from me3cs.framework.data import Data, Index
from me3cs.preprocessing.called import Called


def sort_function_order(func):
    def inner(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._sort_order()
    return inner


class PreprocessingBaseClass:
    """
    Base class for handling preprocessing operations on data.

    Parameters
    ----------
    data : Data or numpy.ndarray
        The input data to be preprocessed. Can be a Data object from the me3cs framework or a numpy ndarray.

    Attributes
    ----------
    data_class : Data
        An instance of the Data class from the me3cs framework for handling data operations.
    called : Called
        An instance of the Called class for storing information about preprocessing functions called.
    data_is_centered : bool
        Indicates whether the data has been mean centered or not.

    Methods
    -------
    reset():
        Resets the preprocessing operations and returns the data to its original state.
    update_is_centered(flag: bool):
        Updates the data_is_centered attribute based on the flag provided.
    call_in_order():
        Calls the preprocessing functions in the correct order.
    """

    def __init__(self, data: [Data, np.ndarray]):

        if isinstance(data, np.ndarray):
            data = Data(data, Index(data.shape[0]), Index(data.shape[1]))
        self.data_class = data

        self.called = Called(list(), list(), list())
        self.data_is_centered = False
        self._reference: [None, np.ndarray] = None

    @property
    def data(self):
        return self.data_class.data

    @data.getter
    def data(self):
        return self.data_class.data

    @data.setter
    def data(self, data):
        self.data_class.preprocessing_data.set(data)

    def update_is_centered(self, flag: bool) -> None:

        setattr(self, "data_is_centered", flag)

    def reset(self) -> None:
        new_data = self.data_class.outlier_detection.get()
        self.data_class.preprocessing_data.set(new_data)
        self.update_is_centered(False)
        self.called.reset()

    def _sort_order(self) -> None:
        # Set the order of the called methods so that is always the last method
        sorted_functions_names = [function.__qualname__ for function in self.called.function]

        called_function_dict = dict(zip(sorted_functions_names, self.called.function))

        sorted_functions_names.sort(key=lambda x: 0 if x.split('.')[0] != "Scaling" else 1)

        sorted_functions = [called_function_dict[function] for function in sorted_functions_names]

        # If the order of the functions are changed, the lists are updated and the methods are called again
        if sorted_functions != self.called.function:
            index = [sorted_functions.index(self.called.function[i]) for i in range(len(self.called.function))]

            self.called.function = sorted_functions
            self.called.args = [new for _, new in sorted(zip(index, self.called.args))]
            self.called.kwargs = [new for _, new in sorted(zip(index, self.called.kwargs))]

            data = self.data_class.get_raw_data()
            self.data_class.preprocessing_data.set(data)

            self.call_in_order()

    def call_in_order(self):
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            function(self, *args, **kwargs)

    def __repr__(self):
        return f"Preprocessing module\n" \
               f"{self.called}"
