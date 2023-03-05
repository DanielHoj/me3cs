from typing import Union

import numpy as np

from cross_validation.cross_validation_split import CrossValidationSplit
from misc.handle_data import transform_array_1d_to_2d
from preprocessing.base import ScalingReference
from preprocessing.called import Called
from preprocessing.filtering import Filtering
from preprocessing.normalisation import Normalisation
from preprocessing.scaling import Scaling
from preprocessing.standardisation import Standardisation


class PreSplitPreprocessing(Normalisation, Filtering, Standardisation):
    def __init__(self, data: np.ndarray, called: Called) -> None:
        super(PreSplitPreprocessing, self).__init__(transform_array_1d_to_2d(data))
        self.called = called
        self.call_in_order()

    def call_in_order(self) -> None:
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            prep_type, function_string = function.__qualname__.split(".")
            if prep_type in ["Normalisation", "Filtering", "Standardisation"]:
                function(self, *args, **kwargs)


class PostSplitPreprocessing(Scaling):
    def __init__(self, data: np.ndarray, reference: np.ndarray, called: Called) -> None:
        super(PostSplitPreprocessing, self).__init__(transform_array_1d_to_2d(data))
        self.reference = ScalingReference(transform_array_1d_to_2d(reference))
        self.called = called

    def call_in_order(self) -> None:
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            prep_type, function_string = function.__qualname__.split(".")
            if prep_type == "Scaling":
                function(self, *args, **kwargs)


class PreprocessingOnSplitData:
    def __init__(
            self,
            split: CrossValidationSplit,
            x_called: Called,
            y_called: Union[None, Called] = None,
    ) -> None:
        self.split = split
        self.x_called = x_called
        self.y_called = y_called
        self.training_set = None
        self.test_set = None
        self.apply_preprocessing()

    def apply_preprocessing(self) -> None:
        split = self.split

        if split.y is not None:
            x_training, y_training = split.training
            x_test, y_test = split.test
        else:
            x_training = split.training[0]
            x_test = split.test[1]

        (
            preprocessed_training_set_x,
            preprocessed_test_set_x,
        ) = apply_preprocessing_on_split_data(
            training_set=x_training, test_set=x_test, called=self.x_called
        )
        if split.y is not None:
            (
                preprocessed_training_set_y,
                preprocessed_test_set_y,
            ) = apply_preprocessing_on_split_data(
                training_set=y_training, test_set=y_test, called=self.y_called
            )
            self.test_set = tuple(
                [preprocessed_training_set_x, preprocessed_training_set_y]
            )
            self.training_set = tuple(
                [preprocessed_test_set_x, preprocessed_test_set_y]
            )
        else:
            self.test_set = tuple([preprocessed_training_set_x])
            self.training_set = tuple([preprocessed_test_set_x])


def apply_preprocessing_on_split_data(
        training_set: list[np.ndarray], test_set: list[np.ndarray], called: Called
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    preprocessed_training_set = []
    preprocessed_test_set = []

    for train, test in zip(training_set, test_set):
        preprocessed_training_set.append(
            PostSplitPreprocessing(
                data=train, reference=train, called=called
            ).data
        )
        preprocessed_test_set.append(
            PostSplitPreprocessing(
                data=test, reference=train, called=called
            ).data
        )
    return preprocessed_training_set, preprocessed_test_set
