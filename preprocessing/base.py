import numpy as np

from framework.helper_classes.base_getter import BaseGetter
from framework.helper_classes.link import Link, create_links, LinkedBranches
from preprocessing.called import Called


class ScalingReference:
    def __init__(self, data: np.ndarray, axis=0):
        self.mean: np.ndarray = data.mean(axis=axis)
        self.std: np.ndarray = data.std(axis=axis)
        self.median: np.ndarray = np.median(data, axis=axis)
        self.sqrt_std: np.ndarray = np.sqrt(data.std(axis=axis))


def sort_function_order(func):
    def inner(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._sort_order()

    return inner


class PreprocessingBaseClass(BaseGetter):
    def __init__(self, data: [list[Link, Link, Link] | np.ndarray], linked_branches: [None, LinkedBranches] = None):
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
        self.reference = ScalingReference(self.data)

    def update_is_centered(self, flag: bool) -> None:
        setattr(self, "data_is_centered", flag)

    def reset(self) -> None:
        if self.linked_branches is not None:
            self.linked_branches.reset_to_link("_missing_data_link")
            self.data = self._missing_data_link.get()
        else:
            self.data = self._missing_data_link.get()
        self.update_is_centered(False)
        self.called = Called(list(), list(), list())

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

            self.data = self._missing_data_link.get()

            for function, args, kwargs in zip(
                    self.called.function, self.called.args, self.called.kwargs
            ):
                function(self, *args, **kwargs)
