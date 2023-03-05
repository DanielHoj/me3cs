import warnings

import numpy as np

from framework.helper_classes.base_getter import BaseGetter
from framework.helper_classes.link import Link, create_links, LinkedBranches
from missing_data.imputation import imputation_algorithms
from missing_data.interpolation import interpolation_algorithms


def check_nan(data: np.ndarray) -> None:
    has_nan = np.isnan(data).any()
    if has_nan:
        warnings.warn("Dataset contain missing values. Consider using the missing_values module.")


class MissingData(BaseGetter):
    def __init__(self, data: [list[Link, Link, Link] | np.ndarray],
                 linked_branches: [LinkedBranches, None] = None) -> None:

        raw_data_link, missing_data_link, preprocessing_data_link, data_link = create_links(data)
        super().__init__(data_link)
        self._raw_data_link = raw_data_link
        self._missing_data_link = missing_data_link
        self._preprocessing_data_link = preprocessing_data_link
        check_nan(self.data)
        self._linked_branches = linked_branches

    def interpolation(self, algorithm: str = "mean") -> None:
        self._check_algorithm_type(algorithm, interpolation_algorithms)
        func = interpolation_algorithms.get(algorithm)
        result = func(self._raw_data_link.get())
        self.data = result
        self._missing_data_link.set(result)

    def imputation(self, algorithm: str = "emsvd") -> None:
        self._check_algorithm_type(algorithm, imputation_algorithms)
        func = imputation_algorithms.get(algorithm)
        result = func(self._raw_data_link.get())
        self.data = result
        self._missing_data_link.set(result)

    def delete_nan(self, dim: int = 0) -> None:
        if dim not in [1, 0]:
            raise ValueError(f'Please input 1 or 0. {dim} was input')

        if dim == 1:
            if np.count_nonzero(~np.isnan(self.data).any(axis=0)) == self.data.shape[1]:
                raise ValueError("No more missing values")
            result = np.delete(self.data, list(np.isnan(self.data).any(axis=0)), 1)

            self.data = result
            self._missing_data_link.set(result)

        else:
            if np.count_nonzero(~np.isnan(self.data).any(axis=1)) == self.data.shape[0]:
                raise ValueError("No more missing values")

            row_nan = ~np.isnan(self.data).any(axis=1)
            self._linked_branches.set_all_rows("_missing_data_link", list(row_nan))

    @staticmethod
    def _check_algorithm_type(algorithm_input: str, algorithm_type: [interpolation_algorithms, imputation_algorithms]):
        algorithm_type = list(algorithm_type.keys())
        if not isinstance(algorithm_input, str):
            raise TypeError(f"Please input a string of {algorithm_type}")
        if algorithm_input not in algorithm_type:
            raise TypeError(f"Please choose one of {algorithm_type}. {algorithm_input} was chosen")

    def reset(self) -> None:
        self._linked_branches.reset_to_link("_raw_data_link")
