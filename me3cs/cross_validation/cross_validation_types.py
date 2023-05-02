from abc import ABC
from dataclasses import dataclass

import numpy as np

from me3cs.misc.handle_data import mask_arr


@dataclass
class CrossValidationFactory(ABC):
    """
    Abstract base class for creating cross-validation data splits.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    n_splits : int, optional
        The number of splits for cross-validation, by default None.
    """
    data: np.ndarray
    n_splits: int = None

    def subset(self):
        pass


@dataclass
class VenetianBlinds(CrossValidationFactory):
    """
    Venetian blinds cross-validation data split.

    Inherits from CrossValidationFactory.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    n_splits : int, optional
        The number of splits for cross-validation, by default None.
    """
    def subset(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        data = self.data.copy()

        training = [
            mask_arr(data, np.arange(i, data.shape[0], self.n_splits))
            for i in range(self.n_splits)
        ]
        test = [data[i:: self.n_splits] for i in range(self.n_splits)]

        return training, test


@dataclass
class ContiguousBlocks(CrossValidationFactory):
    """
    Contiguous blocks cross-validation data split.

    Inherits from CrossValidationFactory.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    n_splits : int, optional
        The number of splits for cross-validation, by default None.
    """
    def subset(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        data = self.data.copy()
        idx = np.arange(data.shape[0])
        msk_training = np.array_split(idx, self.n_splits)

        training = [mask_arr(data, msk) for msk in msk_training]
        test = np.array_split(data, self.n_splits)
        return training, test


@dataclass
class RandomBlocks(CrossValidationFactory):
    """
    Random blocks cross-validation data split.

    Inherits from CrossValidationFactory.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    n_splits : int, optional
        The number of splits for cross-validation, by default None.
    """
    def subset(self) -> list[np.ndarray]:
        data_random = self.data.copy()
        np.random.shuffle(data_random)
        return np.array_split(data_random, self.n_splits)


@dataclass
class Custom(CrossValidationFactory):
    """
    Custom cross-validation data split.

    Inherits from CrossValidationFactory.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    n_splits : int, optional
        The number of splits for cross-validation, by default None.
    """
    def subset(self) -> list[np.ndarray]:
        pass


cross_validation_types = {
    "venetian_blinds": VenetianBlinds,
    "contiguous_blocks": ContiguousBlocks,
    "random_blocks": RandomBlocks,
}
