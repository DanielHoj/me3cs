from dataclasses import dataclass

import numpy as np


@dataclass
class Link:
    data: [int, None]

    def get(self):
        return self.data

    def set(self, data):
        self.data = data


class LinkedBranches:
    def __init__(self, branch: list) -> None:
        self.branches = branch

    def set_rows(self, changed_branch, index_name: str, index: list[bool]) -> None:
        for branch in self.branches:
            if branch is not changed_branch:
                branch._row_index.set_index(index_name, index)

    def get_rows(self) -> None:
        branch = self.branches[0]
        return branch._row_index.__dict__

    def set_all_rows(self, index_name: str, index: list[bool]) -> None:
        self.branches[0]._set_row_index(index_name, index)

    def update_data_from_index(self):
        for branch in self.branches:
            branch._update_data_from_index()

    def add_branch(self, branch):
        self.branches.append(branch)

    def reset_to_link(self, reset_to_link: str):
        for branch in self.branches:
            branch._reset_to(reset_to_link)


def create_links(data: [list[Link, Link, Link, Link] | np.ndarray]) -> tuple[Link, Link, Link, Link]:
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
