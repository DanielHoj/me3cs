import numpy as np

from . import TYPING_CV_STR, cross_validation_types


class CrossValidationSplit:
    """
    Splits the input data into training and test sets for cross-validation.

    Parameters
    ----------
    x : np.ndarray
        Input feature matrix (n_samples, n_features).
    y : np.ndarray
        Output target array (n_samples,).
    percentage_left_out : float
        The percentage of data to leave out for validation during cross-validation.
    cv_type : TYPING_CV_STR
        The type of cross-validation to perform.

    Attributes
    ----------
    x : np.ndarray
        Input feature matrix (n_samples, n_features).
    y : np.ndarray
        Output target array (n_samples,).
    percentage_left_out : float
        The percentage of data to leave out for validation during cross-validation.
    n_splits : int
        The number of splits to perform during cross-validation.
    cv_type : str
        The type of cross-validation to perform.
    test : [None, tuple[..., np.ndarray]]
        Tuple containing the test data split for each fold in cross-validation.
    training : [None, tuple[..., np.ndarray]]
       """
    def __init__(self, x: np.ndarray,
                 y: np.ndarray,
                 percentage_left_out: float,
                 cv_type: TYPING_CV_STR,
                 ) -> None:

        self.x = x
        self.y = y

        self.percentage_left_out = percentage_left_out
        self.n_splits = int(1/percentage_left_out)
        self.cv_type = cv_type

        self.test: [None, tuple[..., np.ndarray]] = None
        self.training: [None, tuple[..., np.ndarray]] = None

        self.split(cv_type)

    @property
    def cv_type(self) -> str:
        """
        Returns the current cross-validation type.
        """
        return self.cv_type

    @cv_type.setter
    def cv_type(self, cv: str) -> None:
        """
        Sets the cross-validation type if it is valid, otherwise raises a ValueError.

        Parameters
        ----------
        cv : str
            The cross-validation type to set.

        Raises
        ------
        ValueError
            If the input cross-validation type is not a valid option.
        """
        if cv not in cross_validation_types.keys():
            raise ValueError(f"Please input {', '.join(cross_validation_types.keys())}. {cv} was input")

    def split(self, cv_type: str) -> None:
        """
        Performs the data splitting for the specified cross-validation type.

        Splits the input data (x, y) into training and test sets according to the
        specified cross-validation type and stores the results in the test and
        training attributes.

        Parameters
        ----------
        cv_type : str
            The type of cross-validation to perform.
        """
        cv = cross_validation_types[cv_type]
        x_training, x_test = cv(self.x, self.n_splits).subset()
        if self.y is not None:
            y_training, y_test = cv(self.y, self.n_splits).subset()
            test = tuple([x_test, y_test])
            training = tuple([x_training, y_training])
        else:
            test = tuple([x_test])
            training = tuple([x_training])

        self.n_splits = len(x_training)
        self.test = test
        self.training = training
