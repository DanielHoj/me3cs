import numpy as np

from me3cs.framework.helper_classes.handle_input import validate_data
from me3cs.framework.helper_classes.link import Link
from me3cs.misc.handle_data import transform_array_1d_to_2d


class Index:
    """
    Class representing the index of rows or columns in the Data class.

    Parameters
    ----------
    length : int
        The number of elements in the index.

    Attributes
    ----------
    missing_data : list[bool]
        A list of boolean values indicating the presence of missing data for each element.
    outlier_detection : list[bool]
        A list of boolean values indicating the presence of outliers for each element.
    """

    _TYPES = ["missing_data", "outlier_detection"]

    def __init__(self, length: int) -> None:
        index = [True for _ in range(length)]
        self.missing_data = index.copy()
        self.outlier_detection = index.copy()

    @property
    def total(self) -> list[..., bool]:
        """
        Returns the merged index of missing data and outlier detection.

        Returns
        -------
        list[bool]
            A list of boolean values representing the merged index.
        """
        return self._merge_index()

    @total.getter
    def total(self) -> list[..., bool]:

        return self._merge_index()

    @property
    def length_of_rows(self) -> int:
        return sum(self.total)

    @length_of_rows.getter
    def length_of_rows(self) -> int:
        return sum(self.total)

    def set_index(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        """
        Updates the specified index (missing_data or outlier_detection) with the given missing_values.

        Parameters
        ----------
        module : str
            The name of the index to update. Must be one of "missing_data" or "outlier_detection".
        missing_values : tuple[int] or int
            The indices of the elements to be marked as missing or outliers.
        """
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
        """
        Resets the specified index (missing_data or outlier_detection) to its initial state.

        Parameters
        ----------
        module : str
            The name of the index to reset. Must be one of "missing_data" or "outlier_detection".
        """
        if module not in self._TYPES:
            raise ValueError(f"module needs to be one of {' or '.join(self._TYPES)}")
        reset_index = [True for _ in range(len(self.total))]
        setattr(self, module, reset_index)

    def reset_all(self) -> None:
        """
        Resets all indices (missing_data and outlier_detection) to their initial states.
        """
        reset_index = [True for _ in range(len(self.total))]
        [setattr(self, module, reset_index) for module in self._TYPES]

    def _merge_index(self) -> list[..., bool]:
        """
        Merges the missing_data and outlier_detection indices into a single list.

        Returns
        -------
        list[bool]
            A list of boolean values representing the merged index.
        """
        merged_list = [a and b for a, b in zip(self.missing_data, self.outlier_detection)]
        merged_list = [False if value is False else True for value in merged_list]
        return merged_list

    def __repr__(self):
        """
        Returns a string representation of the Index instance.

        Returns
        -------
        str
            A string representation of the Index instance, with the length of the rows.
        """
        return f"Index: {self.length_of_rows}"


def check_index(existing: tuple, new: [tuple, int]) -> tuple[int, ...]:
    """
    Adjusts the new indices based on the existing indices. This is useful when you need to
    add new indices to a list without conflicts.

    Parameters
    ----------
    existing : tuple[int]
        A tuple containing the existing indices.
    new : tuple[int] or int
        A tuple or single integer representing the new indices to be added.

    Returns
    -------
    tuple[int]
        A tuple of the adjusted new indices.
    """
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
    """
    Returns the indices of the False elements in a boolean list.

    Parameters
    ----------
    boolean : list[bool]
        A list of boolean values.

    Returns
    -------
    tuple
        A tuple containing the indices of the False elements in the input list.
    """
    return tuple(filter(lambda i: not boolean[i], range(len(boolean))))


class Data:
    """
    Class for handling and storing data.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be preprocessed.
    rows : Index
        The index object for the rows of the data.
    variables : Index
        The index object for the columns of the data.

    Attributes
    ----------
    raw : Link
        An instance of the Link class for the raw data.
    missing_data : Link
        An instance of the Link class for the data with missing data removed.
    preprocessing_data : Link
        An instance of the Link class for the data with preprocessing operations applied.
    outlier_detection : Link
        An instance of the Link class for the data with outliers removed.
    rows : Index
        The index object for the rows of the data.
    variables : Index
        The index object for the columns of the data.
    """

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
        """
        Returns the most processed data.

        Returns
        -------
        numpy.ndarray
            The data.
        """
        return self.preprocessing_data.get()

    @data.getter
    def data(self) -> np.ndarray:
        return self.preprocessing_data.get()

    @data.setter
    def data(self, data) -> None:
        raise TypeError("data cannot be set from here. Please make a new model")

    def remove_rows(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        """
        Removes rows from the specified module based on the missing_values provided.

        Parameters
        ----------
        module : str
            The name of the module to remove rows from.
        missing_values : tuple[int] or int
            The indices of the rows to remove.
        """
        self.rows.set_index(module, missing_values)
        set_data_from_index(self, module)

    def remove_columns(self, module: str, missing_values: [tuple[..., int], int]) -> None:
        """
        Removes columns from the specified module based on the missing_values provided.

        Parameters
        ----------
        module : str
            The name of the module to remove columns from.
        missing_values : tuple[int] or int
            The indices of the columns to remove.
        """
        self.variables.set_index(module, missing_values)
        set_data_from_index(self, module)

    def reset_index(self, module: str) -> None:
        """
        Resets the indices of the specified module.

        Parameters
        ----------
        module : str
            The name of the module to reset indices for.
        """
        if module == "all":
            self.rows.reset_all()
            self.variables.reset_all()
        else:
            self.rows.reset_index(module)
            self.variables.reset_index(module)

        set_data_from_index(self, module)

    def get_raw_data(self) -> np.ndarray:
        """
        Returns the raw data with missing data and outlier detection applied.

        Returns
        -------
        numpy.ndarray
            The raw data with missing data and outlier detection applied.
        """
        data = self.raw.get()
        data = data[self.rows.total, :]
        data = data[:, self.variables.total]
        return data


def set_data_from_index(self, module: str):
    """
    Sets the data attribute of the Data class based on the specified module and dimension.

    Parameters
    ----------
    self : Data
        The Data class instance.
    module : str
        The name of the module to set data from.
    """

    if module == "all":
        module = "missing_data"

    match module:
        case "missing_data":
            data = self.raw.get()

            rows = self.rows.total
            variables = self.variables.total
            new_data = data[np.ix_(rows, variables)]

            for hierarchy in self._HIERARCHY[1:]:
                data_type = getattr(self, hierarchy)
                data_type.set(new_data)

        case "outlier_detection":
            data = self.missing_data.get()

            rows = remove_from_one_list(self.rows.missing_data, self.rows.outlier_detection)
            variables = remove_from_one_list(self.variables.missing_data, self.variables.outlier_detection)
            new_data = data[np.ix_(rows, variables)]

            for hierarchy in self._HIERARCHY[2:]:
                data_type = getattr(self, hierarchy)
                data_type.set(new_data)


def remove_from_one_list(remove: list, keep: list) -> list:
    """
    Removes elements from the 'keep' list based on the boolean values in the 'remove' list.

    Parameters
    ----------
    remove : list[bool]
        A list of boolean values to determine which elements to remove from the 'keep' list.
    keep : list
        A list of elements to be filtered based on the 'remove' list.

    Returns
    -------
    list
        A list with elements removed based on the 'remove' list.
    """
    remove = remove.copy()
    keep = keep.copy()
    for i in range(len(keep)):
        if not remove[i]:
            keep[i] = None
    return [i for i in keep if i is not None]


def missing_values_to_boolean(missing_values: [tuple[..., int], int],
                              old_index: list[..., bool]) -> list[..., bool]:
    """
    Converts missing values to a boolean list, where False represents a missing value and True represents a present
    value.

    Parameters
    ----------
    missing_values : tuple[int] or int
        The indices of the elements marked as missing.
    old_index : list[bool]
        The original boolean list representing the presence of missing values.

    Returns
    -------
    list[bool]
        A new boolean list representing the updated presence of missing values.
    """
    new_index = old_index.copy()
    if isinstance(missing_values, (tuple, list)):
        for i in missing_values:
            new_index[i] = False
    else:
        new_index[missing_values] = False
    return new_index
