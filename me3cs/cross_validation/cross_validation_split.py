import numpy as np

from me3cs.cross_validation import TYPING_CV_STR, cross_validation_types


class CrossValidationSplit:
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
        return self.cv_type

    @cv_type.setter
    def cv_type(self, cv: str) -> None:
        if cv not in cross_validation_types.keys():
            raise ValueError(f"Please input {', '.join(cross_validation_types.keys())}. {cv} was input")

    def split(self, cv_type: str) -> None:
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
