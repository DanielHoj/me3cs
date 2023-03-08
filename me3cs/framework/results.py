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

    def remove_outlier(self, index: [int, tuple[int]]):
        index_total = self._linked_branches.get_rows()["_preprocessing_data_link"].copy()
        if isinstance(index, int):
            index_total[index] = False
        elif isinstance(index, tuple):
            for i in index:
                index_total[i] = False

        self._linked_branches.set_all_rows("_preprocessing_data_link", index_total)
        self._linked_branches.call_preprocessing_in_order()

    def remove_outlier_from_q_residuals(self):
        if self._result.diagnostics is None:
            raise ReferenceError("diagnostics are not calculated")
        elif not hasattr(self._result.diagnostics, "q_residuals"):
            raise ValueError("q_residuals are not calculated")
        q_res = getattr(self._result.diagnostics, "q_residuals")
        q_res_optimal = q_res[:, self._result.optimal_number_component]


    def remove_outlier_from_hotellings_t2(self):
        pass

    def remove_outlier_from_leverage(self):
        pass
