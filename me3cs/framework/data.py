import numpy as np

from me3cs.framework.helper_classes.handle_input import validate_data
from me3cs.framework.helper_classes.link import Link
from me3cs.misc.handle_data import transform_array_1d_to_2d


def missing_values_to_boolean(missing_values: [tuple[..., int], int], old_index: list[..., bool]) -> list[..., bool]:
    new_index = old_index.copy()
    if isinstance(missing_values, (tuple, list)):
        for i in missing_values:
            new_index[i] = False
    else:
        new_index[missing_values] = False
    return new_index


class Index:
    _TYPES = ["missing_data", "outlier_detection"]

    def __init__(self, length: int) -> None:
        index = [True for _ in range(length)]
        self.missing_data = index.copy()
        self.outlier_detection = index.copy()

    @property
    def total(self):
        return self._merge_index()

    @total.getter
    def total(self):
        return self._merge_index()

    @property
    def length_of_rows(self):
        return sum(self.total)

    @length_of_rows.getter
    def length_of_rows(self):
        return sum(self.total)

    def set_index(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        if not isinstance(module, str):
            raise TypeError("The module needs to be a string")
        if module not in self._TYPES:
            raise ValueError(f"module needs to be one of: {', '.join(self._TYPES)}")

        if isinstance(missing_values, int):
            if missing_values >= self.length_of_rows:
                raise ValueError("missing_values is out of bounds")
            updated_missing_values = check_index(count_false(self.total), missing_values)
        else:
            if any(elem >= self.length_of_rows for elem in missing_values):
                raise ValueError("missing_values is out of bounds")
            updated_missing_values = check_index(count_false(self.total), missing_values)
        old_index = getattr(self, module)
        index = missing_values_to_boolean(updated_missing_values, old_index)
        setattr(self, module, index)

    def reset_index(self, module: str) -> None:
        if module not in self._TYPES:
            raise ValueError(f"module needs to be one of {' or '.join(self._TYPES)}")
        reset_index = [True for _ in range(len(self.total))]
        setattr(self, module, reset_index)

    def reset_all(self):
        reset_index = [True for _ in range(len(self.total))]
        [setattr(self, module, reset_index) for module in self._TYPES]

    def _merge_index(self) -> list[..., bool]:
        merged_list = [a and b for a, b in zip(self.missing_data, self.outlier_detection)]
        merged_list = [False if value is False else True for value in merged_list]
        return merged_list

    def __repr__(self):
        return f"Index: {self.length_of_rows}"


def check_index(existing: tuple, new: [tuple, int]) -> tuple[int, ...]:
    result = []
    if isinstance(new, int):
        for j in existing:
            if j <= new:
                new += 1
        result.append(new)
    else:
        for index in new:
            for j in existing:
                # check if a smaller index already exists in the old
                # for each one you find, increment by one
                if j <= index:
                    index += 1
            result.append(index)
    return tuple(result)


def count_false(boolean: list[bool]) -> tuple:
    return tuple(filter(lambda i: not boolean[i], range(len(boolean))))


class Data:
    _HIERARCHY = ["raw", "missing_data", "outlier_detection", "preprocessing_data"]

    def __init__(self, data: np.ndarray, rows: Index, variables: Index) -> None:

        validate_data(data)
        data = transform_array_1d_to_2d(data)
        data = data.astype("float")

        self.raw = Link(data)
        self.missing_data = Link(data)
        self.preprocessing_data = Link(data)
        self.outlier_detection = Link(data)
        self.rows = rows
        self.variables = variables

    @property
    def data(self) -> np.ndarray:
        return self.preprocessing_data.get()

    @data.getter
    def data(self) -> np.ndarray:
        return self.preprocessing_data.get()

    @data.setter
    def data(self, data) -> None:
        raise TypeError("data cannot be set from here. Please make a new model")

    def remove_rows(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        self.rows.set_index(module, missing_values)
        set_data_from_index(self, module, dimension=0)

    def remove_columns(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        self.variables.set_index(module, missing_values)
        set_data_from_index(self, module, dimension=1)

    def reset_index(self, module: str) -> None:
        if module == "all":
            self.rows.reset_all()
            self.variables.reset_all()
        else:
            self.rows.reset_index(module)
            self.variables.reset_index(module)

        set_data_from_index(self, module, dimension=0)
        set_data_from_index(self, module, dimension=1)

    def get_raw_data(self) -> np.ndarray:
        data = self.raw.get()
        data = data[self.rows.total, :]
        data = data[:, self.variables.total]
        return data


def set_data_from_index(self, module, dimension):
    if module == "all":
        module = "missing_data"

    match module:
        case "missing_data":
            data = self.raw.get()
            if dimension == 0:
                index = self.rows.total
                new_data = data[index, :]
            else:
                index = self.variables.total
                new_data = data[:, index]

            for hierarchy in self._HIERARCHY[1:]:
                data_type = getattr(self, hierarchy)
                data_type.set(new_data)

        case "outlier_detection":
            data = self.missing_data.get()
            if dimension == 0:
                index = remove_from_one_list(self.rows.missing_data, self.rows.outlier_detection)
                new_data = data[index, :]
            else:
                index = remove_from_one_list(self.variables.missing_data, self.variables.outlier_detection)
                new_data = data[:, index]
            for hierarchy in self._HIERARCHY[2:]:
                data_type = getattr(self, hierarchy)
                data_type.set(new_data)


def remove_from_one_list(remove: list, keep: list) -> list:
    remove = remove.copy()
    keep = keep.copy()
    for i in range(len(keep)):
        if not remove[i]:
            keep[i] = None
    return [i for i in keep if i is not None]
