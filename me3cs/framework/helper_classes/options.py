class Options:
    def __init__(
        self,
        cross_validation: str = "venetian_blinds",
        n_components: int = 10,
        mean_center: bool = True,
        percentage_left_out: float = 0.1,
    ):
        self.cross_validation = cross_validation
        self.n_components = n_components
        self.mean_center = mean_center
        self.percentage_left_out = percentage_left_out

    def __repr__(self):
        return dict_to_string_with_newline(self.__dict__)

    @property
    def cross_validation(self):
        return self._cross_validation

    @cross_validation.setter
    def cross_validation(self, cv: str):
        cv_options = ["venetian_blinds", "contiguous_blocks", "random_blocks"]
        if cv not in cv_options:
            raise ValueError(f"Please input {', '.join(cv_options)}. {cv} was input")
        self._cross_validation = cv

    @property
    def mean_center(self):
        return self._mean_center

    @mean_center.setter
    def mean_center(self, flag):
        if not isinstance(flag, bool):
            raise TypeError(f"Please input a boolean. {flag} was input.")
        self._mean_center = flag

    @property
    def percentage_left_out(self):
        return self._percentage_left_out

    @percentage_left_out.setter
    def percentage_left_out(self, left_out):
        if not isinstance(left_out, float):
            raise TypeError(
                f"Please input a a float between 0 and 1. {left_out} was input."
            )
        if not (0 < left_out < 1):
            raise ValueError(
                f"Please input a a float between 0 and 1. {left_out} was input."
            )
        self._percentage_left_out = left_out


def dict_to_string_with_newline(d) -> str:
    string_dict = ""
    for key, value in d.items():
        string_dict += str(key).lstrip("_") + ": " + str(value) + "\n"
    return string_dict
