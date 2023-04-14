import numpy as np

from me3cs.framework.results import count_false


class RowIndex:
    def __init__(self, data: np.ndarray) -> None:
        idx_bool = [True for _ in range(data.shape[0])]
        self._raw_data_link = idx_bool.copy()
        self._missing_data_link = idx_bool.copy()
        self._preprocessing_data_link = idx_bool.copy()
        self._data_link = idx_bool.copy()

    @property
    def rows(self):
        return self.get_total_index()

    @rows.getter
    def rows(self):
        return self.get_total_index()

    def set_index(self, index_name: str, index: list[bool]) -> None:
        idx_names = ("_raw_data_link", "_missing_data_link", "_preprocessing_data_link", "_data_link")
        idx_sum = sum(index)
        if index_name not in idx_names:
            raise ValueError(f"input not in {idx_names}")

        match index_name:
            case "_raw_data_link":
                setattr(self, "_raw_data_link", index)
                new_index = [True for _ in range(idx_sum)]
                for idx_name in idx_names[1:]:
                    setattr(self, idx_name, new_index.copy())

            case "_missing_data_link":
                setattr(self, "_missing_data_link", index)
                new_index = [True for _ in range(idx_sum)]
                for idx_name in idx_names[2:]:
                    setattr(self, idx_name, new_index)

            case "_preprocessing_data_link":
                setattr(self, "_preprocessing_data_link", index)
                new_index = [True for _ in range(idx_sum)]
                for idx_name in idx_names[3:]:
                    setattr(self, idx_name, new_index)

            case "_data_link":
                setattr(self, "_data_link", index)

    def get_index(self, index_name: str) -> list[bool]:
        idx_names = ("_raw_data_link", "_missing_data_link", "_preprocessing_data_link", "_data_link")
        if index_name not in idx_names:
            raise ValueError(f"input not in {idx_names}")

        index = getattr(self, index_name)
        return index

    def get_total_index(self) -> list[bool]:
        rows = self.__dict__.values()
        row_idx = list()

        for row in rows:
            row_idx.append(count_false(row))
        added_row_idx = add_all_idx(row_idx)
        bool_idx = self._raw_data_link.copy()
        for i in added_row_idx:
            bool_idx[i] = False

        return bool_idx


def add_idx(existing: tuple, new: tuple) -> list:
    c = []
    for element in new:
        count = sum(1 for x in existing if x <= element)
        c.append(count + element)
    c.extend(existing)
    c.sort()
    return c


def add_all_idx(index_rows: list[tuple]) -> list:
    total_idx = []
    for i, idx in enumerate(index_rows):
        new_idx = add_idx(tuple(total_idx), idx)
        total_idx = new_idx
    return total_idx

