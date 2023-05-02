import numpy as np

from me3cs.cross_validation.cross_validation_split import CrossValidationSplit
from me3cs.misc.handle_data import transform_array_1d_to_2d
from me3cs.preprocessing.called import Called
from me3cs.preprocessing.filtering import Filtering
from me3cs.preprocessing.normalisation import Normalisation
from me3cs.preprocessing.scaling import Scaling
from me3cs.preprocessing.standardisation import Standardisation


class PreSplitPreprocessing(Normalisation, Filtering, Standardisation):
    """
    Applies non-scaling preprocessing methods on the input data before splitting it for cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input data to preprocess.
    called : Called
        The preprocessing methods to apply on the data.
    """
    def __init__(self, data: np.ndarray, called: Called) -> None:
        super(PreSplitPreprocessing, self).__init__(transform_array_1d_to_2d(data))
        self.called = called
        self.call_in_order()

    def call_in_order(self) -> None:
        """
        Applies the preprocessing methods in the order specified by the called attribute.
        """
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            prep_type, function_string = function.__qualname__.split(".")
            if prep_type in ["Normalisation", "Filtering", "Standardisation"]:
                function(self, *args, **kwargs)


class PostSplitPreprocessing(Scaling):
    """
    Applies scaling preprocessing methods on the input data after splitting it for cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input data to preprocess.
    reference : np.ndarray
        The reference data used to determine the parameters for the preprocessing methods.
    called : Called
        The preprocessing methods to apply on the data.
    """
    def __init__(self, data: np.ndarray, reference: np.ndarray, called: Called) -> None:
        super(PostSplitPreprocessing, self).__init__(transform_array_1d_to_2d(data))
        self.called = called
        self._reference = reference

    def call_in_order(self) -> None:
        """
        Applies the preprocessing methods in the order specified by the called attribute.
        """
        for function, args, kwargs in zip(
                self.called.function, self.called.args, self.called.kwargs
        ):
            prep_type, function_string = function.__qualname__.split(".")
            if prep_type == "Scaling":
                function(self, *args, **kwargs)


class PreprocessingOnSplitData:
    """
    Applies the specified preprocessing methods on the split data for cross-validation.

    Parameters
    ----------
    split : CrossValidationSplit
        The split data for cross-validation.
    x_called : Called
        The preprocessing methods to apply on the input feature matrix.
    y_called : [None, Called], optional, default=None
        The preprocessing methods to apply on the output target array, if any.

    Attributes
    ----------
    split : CrossValidationSplit
        The split data for cross-validation.
    x_called : Called
        The preprocessing methods to apply on the input feature matrix.
    y_called : [None, Called]
        The preprocessing methods to apply on the output target array, if any.
    training_set : None, tuple[list[np.ndarray], list[np.ndarray]]
        Tuple containing the lists of preprocessed training input data (x_training) and output
        data (y_training), if any.
    test_set : None, tuple[list[np.ndarray], list[np.ndarray]]
        Tuple containing the lists of preprocessed test input data (x_test) and output
        data (y_test), if any.
    """
    def __init__(
            self,
            split: CrossValidationSplit,
            x_called: Called,
            y_called: [None, Called] = None,
    ) -> None:
        self.split = split
        self.x_called = x_called
        self.y_called = y_called
        self.training_set = None
        self.test_set = None
        self.apply_preprocessing()

    def apply_preprocessing(self) -> None:
        """
        Applies the specified preprocessing methods on the split data.
        """
        split = self.split

        x_training, y_training = split.training
        x_test, y_test = split.test

        (
            preprocessed_training_set_x,
            preprocessed_test_set_x,
        ) = apply_preprocessing_on_split_data(
            training_set=x_training, test_set=x_test, called=self.x_called
        )
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


def apply_preprocessing_on_split_data(
        training_set: list[np.ndarray, ...], test_set: list[np.ndarray, ...], called: Called
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Applies the specified preprocessing methods on the training and test sets.

    Parameters
    ----------
    training_set : list[np.ndarray, ...]
        The list of training input data.
    test_set : list[np.ndarray, ...]
        The list of test input data.
    called : Called
        The preprocessing methods to apply on the data.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        A tuple containing the lists of preprocessed training and test input data.
    """
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
