import numpy as np

from me3cs.framework.helper_classes.base_getter import BaseGetter
from me3cs.framework.helper_classes.link import Link, create_links, LinkedBranches
from me3cs.preprocessing.called import Called


class ScalingReference:
    """
    Class for computing scaling reference values for data.

    Parameters
    ----------
    data : np.ndarray
        Data for which scaling reference values are to be computed.
    axis : int, optional
        Axis along which to compute reference values. Default is 0.

    Attributes
    ----------
    mean : np.ndarray
        Mean of the data along the specified axis.
    std : np.ndarray
        Standard deviation of the data along the specified axis.
    median : np.ndarray
        Median of the data along the specified axis.
    sqrt_std : np.ndarray
        Square root of the standard deviation of the data along the specified axis.
    """
    def __init__(self, data: np.ndarray, axis=0):
        self.mean: np.ndarray = data.mean(axis=axis)
        self.std: np.ndarray = data.std(axis=axis)
        self.median: np.ndarray = np.median(data, axis=axis)
        self.sqrt_std: np.ndarray = np.sqrt(data.std(axis=axis))


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


class PreprocessingBaseClass(BaseGetter):
    """
    Base class for preprocessing data. Inherits from `BaseGetter`.

    Parameters
    ----------
    data : list[Link, Link, Link] or np.ndarray
        The input data to be preprocessed.
    linked_branches : None or LinkedBranches, optional
        Linked branches object. Default is `None`.

    Attributes
    ----------
    _raw_data_link : Link
        A link to the raw data.
    _missing_data_link : Link
        A link to the missing data.
    preprocessing_data_link : Link
        A link to the preprocessed data.
    called : Called
        Object to keep track of the functions that have been called.
    data_is_centered : bool
        Flag to indicate if the data has been centered.
    reference : ScalingReference
        Object containing information about scaling reference.

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
    def __init__(self, data: [list[Link, Link, Link] | np.ndarray], linked_branches: [None, LinkedBranches] = None):
        """
        Initialize the `PreprocessingBaseClass` object.
        """
        raw_data_link, missing_data_link, preprocessing_data_link, data_link = create_links(data)
        if linked_branches is not None:
            self.linked_branches = linked_branches

        super().__init__(data_link)
        self._raw_data_link = raw_data_link
        self._missing_data_link = missing_data_link
        self.preprocessing_data_link = preprocessing_data_link

        self.called = Called(list(), list(), list())
        self.data_is_centered = False
        self.reference = ScalingReference(self.data)

    def set_ref(self):
        """
        Set the reference for scaling.
        """
        self.reference = ScalingReference(self.data)

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
        if self.linked_branches is not None:
            self.linked_branches.reset_to_link("_missing_data_link")
            self.data = self._missing_data_link.get()
        else:
            self.data = self._missing_data_link.get()
        self.update_is_centered(False)
        self.called = Called(list(), list(), list())

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

            self.data = self._missing_data_link.get()

            for function, args, kwargs in zip(
                    self.called.function, self.called.args, self.called.kwargs
            ):
                function(self, *args, **kwargs)
