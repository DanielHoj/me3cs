from me3cs.framework.helper_classes.link import LinkedBranches


class OutlierDetection:
    def __init__(self, linked_branches: LinkedBranches):
        self._linked_branches = linked_branches

    def remove_outlier(self, index: [int, tuple[int]]):
        index_total = self._linked_branches.get_rows()["_preprocessing_data_link"].copy()
        if isinstance(index, int):
            index_total[index] = False
        elif isinstance(index, tuple):
            for i in index:
                index_total[i] = False

        self._linked_branches.set_all_rows("_preprocessing_data_link", index_total)
        self._linked_branches.call_preprocessing_in_order()


class Results:
    def __init__(self, linked_branches: LinkedBranches):
        self.calibration = None
        self.cross_validation = None
        self.diagnostics = None
        self.outlier_detection = OutlierDetection(linked_branches)

