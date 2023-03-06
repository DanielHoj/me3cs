import numpy as np
import pandas as pd

from me3cs.framework.helper_classes.base_getter import BaseGetter
from me3cs.framework.helper_classes.handle_input import validate_data, save_column_and_index, to_pandas, \
    get_preprocessing_from_dimension
from me3cs.framework.helper_classes.link import create_links, LinkedBranches
from me3cs.misc.handle_data import transform_array_1d_to_2d
from me3cs.missing_data.core import MissingData
from me3cs.preprocessing.called import Called


class RowIndex:
    def __init__(self, data: np.ndarray) -> None:
        idx_bool = [True for _ in range(data.shape[0])]
        self._raw_data_link = idx_bool.copy()
        self._missing_data_link = idx_bool.copy()
        self._preprocessing_data_link = idx_bool.copy()
        self._data_link = idx_bool.copy()

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


class Branch(BaseGetter):
    def __init__(self, data: [np.ndarray, pd.Series, pd.DataFrame], linked_branches: LinkedBranches) -> None:
        validate_data(data)
        preprocessing_type = get_preprocessing_from_dimension(data)
        self._pandas_data = to_pandas(data)
        self._attributes = save_column_and_index(data)
        data = transform_array_1d_to_2d(data)
        data = data.astype("float")

        self._linked_branches = linked_branches
        self._row_index = RowIndex(data)

        data_links = create_links(data)
        self.preprocessing = preprocessing_type(data_links, self._linked_branches)
        self.missing_data = MissingData(data_links, self._linked_branches)
        self._raw_data_link, self._missing_data_link, self._preprocessing_data_link, self._data_link = data_links
        super().__init__(self._data_link)

    def _set_linked_branches(self, linked_branches: LinkedBranches) -> None:
        self._linked_branches = linked_branches

    def _set_row_index(self, index_name: str, index: list[bool]) -> None:
        self._row_index.set_index(index_name, index)
        if self._linked_branches is not None:
            self._linked_branches.set_rows(self, index_name, index)
            self._linked_branches.update_data_from_index()

    def _update_data_from_index(self) -> None:
        link_names = ("_raw_data_link", "_missing_data_link", "_preprocessing_data_link", "_data_link")

        for i, link_name in enumerate(link_names):
            index = getattr(self._row_index, link_name)

            for key in link_names[i:]:
                link = getattr(self, key)
                data = link.get()
                link.set(data[index])

    def _reset_to(self, reset_to_link: str) -> None:
        link_names = ("_raw_data_link", "_missing_data_link", "_preprocessing_data_link", "_data_link")

        if reset_to_link not in link_names:
            raise ValueError(f"Please input one of {link_names}. {reset_to_link} was input")

        match reset_to_link:
            case "_raw_data_link":
                self.preprocessing.called = Called(list(), list(), list())
                self.preprocessing.update_is_centered(False)
                row_idx = self._row_index.get_index(reset_to_link)
                self._row_index.set_index(reset_to_link, row_idx)
                data = getattr(self, reset_to_link).get()
                for name in link_names:
                    link = getattr(self, name)
                    link.set(data)

            case "_missing_data_link":
                self.preprocessing.called = Called(list(), list(), list())
                self.preprocessing.update_is_centered(False)
                row_idx = self._row_index.get_index(reset_to_link)
                self._row_index.set_index(reset_to_link, row_idx)
                data = getattr(self, reset_to_link).get()
                for name in link_names[1:]:
                    link = getattr(self, name)
                    link.set(data)

            case "_preprocessing_data_link":
                row_idx = self._row_index.get_index(reset_to_link)
                self._row_index.set_index(reset_to_link, row_idx)
                data = getattr(self, reset_to_link).get()
                for name in link_names[2:]:
                    link = getattr(self, name)
                    link.set(data)

            case "_data_link":
                row_idx = self._row_index.get_index(reset_to_link)
                self._row_index.set_index(reset_to_link, row_idx)
                data = getattr(self, reset_to_link).get()
                for name in link_names[3:]:
                    link = getattr(self, name)
                    link.set(data)
