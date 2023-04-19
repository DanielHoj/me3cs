import numpy as np

from me3cs.framework.helper_classes.handle_input import validate_data
from me3cs.framework.helper_classes.link import Link


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
    def index(self):
        return self._merge_index()

    @property
    def length_of_rows(self):
        return sum(self.index)

    @length_of_rows.getter
    def length_of_rows(self):
        return sum(self.index)

    def set_index(self, module: str, missing_values: [tuple[..., int], int]) -> None:

        if not isinstance(module, str):
            raise TypeError("The module needs to be a string")
        if module not in self._TYPES:
            raise ValueError(f"module needs to be one of: {', '.join(self._TYPES)}")

        if isinstance(missing_values, int):
            if missing_values >= self.length_of_rows:
                raise ValueError("missing_values is out of bounds")
            updated_missing_values = check_index(count_false(self.index), missing_values)
        else:
            if any(elem >= self.length_of_rows for elem in missing_values):
                raise ValueError("missing_values is out of bounds")
            updated_missing_values = check_index(count_false(self.index), missing_values)
        old_index = getattr(self, module)
        index = missing_values_to_boolean(updated_missing_values, old_index)
        setattr(self, module, index)

    def reset_index(self, module: str) -> None:
        if module not in self._TYPES:
            raise ValueError(f"module needs to be one of {' or '.join(self._TYPES)}")
        reset_index = [True for _ in range(len(self.index))]
        setattr(self, module, reset_index)

    def _merge_index(self) -> list[..., bool]:
        merged_list = [a and b for a, b in zip(self.missing_data, self.outlier_detection)]
        merged_list = [False if value is False else True for value in merged_list]
        return merged_list


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
    _HIERARCHY = ["raw", "missing_data", "preprocessing_data"]

    def __init__(self, data: np.ndarray, rows: Index, variables: Index) -> None:
        validate_data(data)
        self.raw = Link(data)
        self.missing_data = Link(data)
        self.preprocessing_data = Link(data)
        self.rows = rows
        self.variables = variables

    def set_rows(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        self.rows.set_index(module, missing_values)
        data = self.raw.get()
        index = self.rows.total
        new_data = data[index]
        if module == "missing_data":
            for hierarchy in self._HIERARCHY[1:]:
                data_type = getattr(self, hierarchy)
                data_type.set(new_data)
        else:
            self.preprocessing_data.set(new_data)

    def set_variables(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        pass

    def get_raw_data(self) -> np.ndarray:
        data = self.raw.get()
        index = self.rows.total
        return data[index]
