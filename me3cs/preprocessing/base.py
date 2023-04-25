import numpy as np

from me3cs.framework.data import Data, Index
from me3cs.framework.helper_classes.link import Link
from me3cs.preprocessing.called import Called


def sort_function_order(func):
    """
    A decorator that ensures that the _sort_order() method is called after the decorated method.

    Parameters
    ----------
    func : callable
        The function being decorated.

    Returns
    -------
    callable
        A new function that wraps the original function and calls _sort_order() afterwards.
    """

    def inner(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._sort_order()

    return inner


class PreprocessingBaseClass:
    """
    Base class for preprocessing data. Inherits from `BaseGetter`.

    Parameters
    ----------
    data : list[Link, Link, Link] or np.ndarray
        The input data to be preprocessed.


    Attributes
    ----------
    called : Called
        Object to keep track of the functions that have been called.
    data_is_centered : bool
        Flag to indicate if the data has been centered.

    Methods
    -------
    set_ref()
        Set the reference for scaling.
    update_is_centered(flag: bool) -> None
        Update `data_is_centered` attribute with the provided flag.
    reset() -> None
        Reset the object to its original state.
    _sort_order() -> None
        Sort the called methods in order of their execution.

    Notes
    -----
    This class is used as a base class for preprocessing classes. It provides methods for resetting, updating and
    sorting the called methods. It also provides attributes for keeping track of the state of the object.
    """

    def __init__(self, data: [Data, np.ndarray]):
        """
        Initialize the `PreprocessingBaseClass` object.
        """
        if isinstance(data, np.ndarray):
            data = Data(data, Index(data.shape[0]), Index(data.shape[1]))
        self.data_class = data

        self.called = Called(list(), list(), list())
        self.data_is_centered = False

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
        """
        Update `data_is_centered` attribute with the provided flag.

        Parameters
        ----------
        flag : bool
            The flag to set for `data_is_centered`.
        """
        setattr(self, "data_is_centered", flag)

    def reset(self) -> None:
        """
        Reset the object to its original state.
        """
        self.data_class.reset_index("outlier_detection")
        self.update_is_centered(False)
        self.called.reset()

    def _sort_order(self) -> None:
        """
        Sort the called methods in order of their execution. Ensures that methods from the 'Scaling'
        module is called last
        """
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
