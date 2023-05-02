class Options:
    """
    A class to store configuration options for the me3cs model.

    Parameters
    ----------
    cross_validation : str, optional
        The cross-validation method, default is 'venetian_blinds'.
    n_components : int, optional
        The number of components, default is 10.
    mean_center : bool, optional
        Whether to mean-center the data, default is True.
    percentage_left_out : float, optional
        The percentage of data to be left out in cross-validation, default is 0.1.

    Attributes
    ----------
    cross_validation : str
        The cross-validation method.
    n_components : int
        The number of components.
    mean_center : bool
        Whether to mean-center the data.
    percentage_left_out : float
        The percentage of data to be left out in cross-validation.
    """

    def __init__(
        self,
        cross_validation: str = "venetian_blinds",
        n_components: int = 10,
        mean_center: bool = True,
        percentage_left_out: float = 0.1,
    ) -> None:
        self.cross_validation = cross_validation
        self.n_components = n_components
        self.mean_center = mean_center
        self.percentage_left_out = percentage_left_out

    def __repr__(self) -> str:
        """
        Return a string representation of all value in the Options object.

        Returns
        -------
        str
            A string representation of all value in the Options object.
        """
        return dict_to_string_with_newline(self.__dict__)

    @property
    def cross_validation(self) -> str:
        """
        Get the cross-validation method.

        Returns
        -------
        str
            The cross-validation method.
        """
        return self._cross_validation

    @cross_validation.setter
    def cross_validation(self, cv: str) -> None:
        """
        Set the cross-validation method.

        Parameters
        ----------
        cv : str
            The cross-validation method to be set.

        Raises
        ------
        ValueError
            If the input cross-validation method is not valid.
        """
        cv_options = ["venetian_blinds", "contiguous_blocks", "random_blocks"]
        if cv not in cv_options:
            raise ValueError(f"Please input {', '.join(cv_options)}. {cv} was input")
        self._cross_validation = cv

    @property
    def mean_center(self) -> bool:
        """
        Get the mean-center flag.

        Returns
        -------
        bool
            The mean-center flag.
        """
        return self._mean_center

    @mean_center.setter
    def mean_center(self, flag: bool) -> None:
        """
        Set the mean-center flag.

        Parameters
        ----------
        flag : bool
            The mean-center flag to be set.

        Raises
        ------
        TypeError
            If the input flag is not a boolean.
        """
        if not isinstance(flag, bool):
            raise TypeError(f"Please input a boolean. {flag} was input.")
        self._mean_center = flag

    @property
    def percentage_left_out(self) -> float:
        """
        Get the percentage of data to be left out in cross-validation.

        Returns
        -------
        float
            The percentage of data to be left out in cross-validation.
        """
        return self._percentage_left_out

    @percentage_left_out.setter
    def percentage_left_out(self, left_out):
        """
        Set the percentage of data to be left out in cross-validation as a floating point number between 0 and 1.

        Parameters
        ----------
        left_out : float
            The percentage of data to be left out in cross-validation.

        Raises
        ------
        TypeError
            If the input value is not a float.
        ValueError
            If the input value is not between 0 and 1.
        """
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
    """
    Convert a dictionary to a string with each key-value pair on a new line.

    Parameters
    ----------
    d : dict
        The dictionary to be converted.

    Returns
    -------
    str
        A string representation of the input dictionary with each key-value pair on a new line.
    """
    string_dict = ""
    for key, value in d.items():
        string_dict += str(key).lstrip("_") + ": " + str(value) + "\n"
    return string_dict
