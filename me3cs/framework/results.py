import numpy as np

from me3cs.framework.helper_classes.link import LinkedBranches


class Results:
    def __init__(self, linked_branches: LinkedBranches) -> None:
        self.calibration = None
        self.cross_validation = None
        self.diagnostics = None
        self.outlier_detection = OutlierDetection(self, linked_branches)
        self.optimal_number_component = None


class OutlierDetection:
    def __init__(self, result: Results, linked_branches: LinkedBranches):
        self._linked_branches = linked_branches
        self._result = result

    def remove_outlier(self, index: [int, list[int], tuple[int], np.ndarray]):
        index_total = self._linked_branches.get_rows()["_preprocessing_data_link"].copy()
        false_index = count_false(index_total)
        if isinstance(index, int):
            updated_index = index_checker_int(false_index, index)
            index_total[updated_index] = False
        elif isinstance(index, (tuple, list, np.ndarray)):
            updated_index = index_checker_tuple(false_index, index)
            for i in updated_index:
                index_total[i] = False
        else:
            raise TypeError("Input should be an integer or a tuple")
        self._linked_branches.set_all_rows("_preprocessing_data_link", index_total)
        self._linked_branches.call_preprocessing_in_order()

    def _remove_outlier_from(self, number_of_outliers_to_remove: int, diagnostic_name: str):
        if self._result.diagnostics is None:
            raise ReferenceError("diagnostics are not calculated")
        elif not hasattr(self._result.diagnostics, diagnostic_name):
            raise ValueError(f"{diagnostic_name} is not calculated")

        if not isinstance(self._result.optimal_number_component, int):
            raise ValueError("optimal number of components must be chosen")

        diagnostic = getattr(self._result.diagnostics, diagnostic_name)
        diagnostic_optimal = diagnostic[:, self._result.optimal_number_component]
        diagnostic_optimal = diagnostic_optimal.argsort()
        outliers_to_remove = diagnostic_optimal[-number_of_outliers_to_remove:]
        self.remove_outlier(outliers_to_remove)

    def remove_outlier_from_q_residuals(self, number_of_outliers_to_remove: int = 1):
        self._remove_outlier_from(number_of_outliers_to_remove, "q_residuals")

    def remove_outlier_from_hotellings_t2(self, number_of_outliers_to_remove: int = 1):
        self._remove_outlier_from(number_of_outliers_to_remove, "hotelling_t2")

    def remove_outlier_from_leverage(self, number_of_outliers_to_remove: int = 1):
        self._remove_outlier_from(number_of_outliers_to_remove, "leverage")

    def reset(self):
        self._linked_branches.reset_to_link("_missing_data_link")


def index_checker_tuple(existing: tuple, new: tuple) -> list:
    c = []
    for element in new:
        count = sum(1 for x in existing if x <= element)
        c.append(count + element)
    return c


def index_checker_int(existing: tuple, new: int) -> int:
    return sum(1 for x in existing if x <= new) + new


def count_false(boolean: list[bool]) -> tuple:
    return tuple(filter(lambda i: not boolean[i], range(len(boolean))))
