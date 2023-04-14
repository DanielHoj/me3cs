from dataclasses import dataclass

import numpy as np


@dataclass
class Link:
    """
    A dataclass for storing a single piece of data. Mainly used as a link between the modules in the branch class.

    Parameters
    ----------
    data : {np.ndarray, None}
        The data to be stored.

    Examples
    --------
    >>> link = Link(5)
    >>> link.get()
    5
    >>> link.set(10)
    >>> link.get()
    10
    """
    data: [np.ndarray, None]

    def get(self):
        """
        Returns the data stored in the Link object.

        Returns
        -------
        {np.ndarray, None}
            The data stored in the Link object.
        """
        return self.data

    def set(self, data):
        """
        Sets the data stored in the Link object to the input data.

        Parameters
        ----------
        data : {np.ndarray, None}
            The data to be stored in the Link object.
        """
        self.data = data


class LinkedBranches:
    def __init__(self, branch: list) -> None:
        """
        Initialize a LinkedBranches instance.

        Parameters:
        -----------
        branch: list
            A list of linked branches to be collected.
        """
        self.branches = branch

    def set_rows(self, changed_branch, index_name: str, index: list[bool]) -> None:
        """
        Set the row index of all the branches except the changed branch.

        Parameters:
        -----------
        changed_branch: Branch
            The branch that was changed.
        index_name: str
            The name of the index column to be set.
        index: list[bool]
            A boolean list specifying the rows to select.
        """
        for branch in self.branches:
            if branch is not changed_branch:
                branch._row_index.set_index(index_name, index)

    def get_rows(self) -> dict:
        """
        Return a dictionary containing the current row index of the first branch.
        """
        branch = self.branches[0]
        return branch._row_index.__dict__

    def set_all_rows(self, index_name: str, index: list[bool]) -> None:
        """
        Set the row index of all the branches to the same values.

        Parameters:
        -----------
        index_name: str
            The name of the index column to be set.
        index: list[bool]
            A boolean list specifying the rows to select.
        """
        self.branches[0]._set_row_index(index_name, index)

    def update_data_from_index(self) -> None:
        """
        Update the data of all branches from their respective row indices.
        """
        for branch in self.branches:
            branch._update_data_from_index()

    def add_branch(self, branch) -> None:
        """
        Add a new branch to the linked branches.

        Parameters:
        -----------
        branch: Branch
            A branch to be added.
        """
        self.branches.append(branch)

    def reset_to_link(self, reset_to_link: str) -> None:
        """
        Reset all branches to the given link.

        Parameters:
        -----------
        reset_to_link: str
            The name of the link to reset all branches to.
        """
        for branch in self.branches:
            branch._reset_to(reset_to_link)

    def call_preprocessing_in_order(self):
        for branch in self.branches:
            branch.preprocessing.call_in_order()


def create_links(data: [list[Link, Link, Link, Link] | np.ndarray]) -> tuple[Link, Link, Link, Link]:
    """Creates links for the 'raw', 'missing', 'preprocessing' and 'data' arrays.

    Parameters:
    -----------
    data : [list[tuple[Link, Link, Link, Link]], np.ndarray]
        The data to be linked. Can be a list of tuples containing the Link instances or a numpy array.

    Returns:
    --------
    tuple[Link, Link, Link, Link]
        A tuple containing the raw, missing, preprocessing and data links.

    Raises:
    -------
    TypeError:
        If the input data is not a list of tuples or a numpy array.

    Notes:
    ------
    This function creates Link instances for each of the arrays (raw, missing, preprocessing and data),
    which allows the arrays to be linked together and any changes made to one array will reflect on all linked arrays.
    If the data parameter is a np.ndarray, it creates the links.

    """
    if isinstance(data, (list, tuple)):
        raw_data_link, missing_data_link, preprocessing_data_link, data_link = data
    elif isinstance(data, np.ndarray):
        raw_data_link = Link(data)
        missing_data_link = Link(data)
        preprocessing_data_link = Link(data)
        data_link = Link(data)
    else:
        raise TypeError('Please input numpy array or a list of np.ndarray')
    return raw_data_link, missing_data_link, preprocessing_data_link, data_link
