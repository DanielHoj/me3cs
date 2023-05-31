from __future__ import annotations

import datetime
import typing
from copy import deepcopy

import pandas as pd

from me3cs.framework.branch import Branch
from me3cs.framework.data import Data
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.results import Results
from me3cs.preprocessing.base import ScalingAttributes
from me3cs.preprocessing.called import Called

if typing.TYPE_CHECKING:
    from me3cs.framework.base_model import BaseModel


class LogObject:
    """
    A class that represent the current state of the me3cs model. This includes the results, options,
    type of preprocessing and missing data actions performed for both branches. The LogObject works by reference and
    when a new entry is made in the Log, a deepcopy of its attributes is created.
    The parameters include the metadata representing the current state of the me3cs model.

    Parameters
    ----------
    prep : tuple[Called, ...]
        A tuple containing the called preprocessing methods for each branch as a Called object.
    missing_data : tuple[Called, ...]
        A tuple containing the called missing data methods for each branch as a Called object.
    data : tuple[Data, ...]
        A tuple containing Data objects for each branch.
    results : Results
        A Results object containing the results of the model.
    options : Options
        An Options object containing the options for the model.

    Attributes
    ----------
    x_prep : Called or tuple[Called, Called]
         Called preprocessing methods for the x branch represented as a Called object.
    y_prep : Called or tuple[Called, Called], optional
        Called preprocessing methods for the y branch represented as a Called object.
    x_missing : Called
        Called missing data methods for the x branch represented as a Called object
    y_missing : Called, optional
        Called missing data methods for the y branch represented as a Called object.
    x_data : Data
        Data object for the x branch.
    y_data : Data, optional
        Data object for the y branch.
    results : Results
        A Results object containing the results of the model.
    options : Options
        An Options object containing the options for the model.
    prep : tuple[Called, ...]
        A tuple containing the called preprocessing methods for each branch as a Called object.
    missing_data : tuple[Called, ...]
        A tuple containing the called missing data methods for each branch as a Called object.
    data : tuple[Data, ...]
        A tuple containing Data objects for each branch.
    rows : int
        The total number of rows in the data.
    variables : int
        The total number of variables in the x data.
    created_at : datetime.datetime
        The creation time of the LogObject instance.
    last_model_called : str, optional
        The name of the last model called.
    comment : str, optional
        A user-defined comment for the log entry.

    Methods
    -------
    __repr__()
        Returns a string representation of the LogObject instance.
    __copy__()
        Returns a shallow copy of the LogObject instance.
    __deepcopy__(memo)
        Returns a deep copy of the LogObject instance.
    add_comment(comment: [None, str])
        Adds a user-defined comment to the log entry.
    """
    def __init__(self,
                 prep: tuple[Called, ...],
                 prep_attr: tuple[ScalingAttributes, ...],
                 missing_data: tuple[Called, ...],
                 data: tuple[Data, ...],
                 results: Results,
                 options: Options,
                 ) -> None:

        if len(prep) > 0:
            self.x_prep, self.y_prep = prep
            self.x_prep_attr, self.y_prep_attr = prep_attr
            self.x_missing, self.y_missing = missing_data
            self.x_data, self.y_data = data
        else:
            self.x_prep = prep[0]
            self.x_prep_attr = prep_attr[0]
            self.x_missing = missing_data[0]
            self.x_data = data[0]

        self.results = results
        self.options = options

        self.prep = prep
        self.prep_attr = prep_attr
        self.missing_data = missing_data
        self.data = data
        self.rows = data[0].rows.total
        self.variables = data[0].variables.total

        self.created_at = datetime.datetime.now().replace(microsecond=0)
        self.last_model_called = None
        self.comment = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the LogObject instance.

        Returns
        -------
        str
            A string representation of the LogObject instance.
        """
        return f"model type: {self.last_model_called} - created {self.created_at}"

    def __copy__(self) -> LogObject:
        """
        Returns a shallow copy of the LogObject instance.

        Returns
        -------
        LogObject
            A shallow copy of the LogObject instance.
        """
        new_obj = LogObject(self.prep, self.prep_attr, self.missing_data, self.data, self.results, self.options)
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def __deepcopy__(self, memo) -> LogObject:
        """
        Returns a deep copy of the LogObject instance.

        Parameters
        ----------
        memo : dict
            A dictionary to memoize objects for deep copying.

        Returns
        -------
        LogObject
            A deep copy of the LogObject instance.
        """
        new_obj = LogObject(deepcopy(self.prep, memo), deepcopy(self.prep_attr),
                            deepcopy(self.missing_data, memo),
                            deepcopy(self.data, memo), deepcopy(self.results, memo),
                            deepcopy(self.options, memo),
                            )
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def add_comment(self, comment: [None, str]) -> None:
        """
        Adds a user-defined comment to the log entry.

        Parameters
        ----------
        comment : str, optional
            A user-defined comment for the log entry.

        Raises
        ------
        TypeError
            If the input comment is not a string or None.
        """
        if not (isinstance(comment, str) or comment is None):
            raise TypeError("comment needs to be a string")
        self.comment = comment


class Log:
    """
    A class that logs the entries of a me3cs model at a specific state. This includes the results, options,
    type of preprocessing and missing data actions performed for both branches. At any given time can the model be
    reverted back to the desired state through the set_model_from_log method.

    Parameters
    ----------
    model : BaseModel
        The BaseModel object for the model.
    results : Results
        A Results object storing model results.
    options : Options
        An Options object storing model options.

    Attributes
    ----------
    branches : list
        A list of Branch objects for each data array.
    _model : BaseModel
        The BaseModel object for the model.
    log_object : LogObject
        The current LogObject for the model.
    entries : list
        A list of LogObject entries. These are made with deepcopy, and has no reference to the current model state.
    """
    def __init__(self, model: "BaseModel", results: Results,
                 options: Options):

        self.branches = model.branches
        prep = tuple(prep.preprocessing.called for prep in self.branches)
        prep_attr = tuple(prep.preprocessing.scaling_attributes for prep in self.branches)
        missing_data = tuple(missing.missing_data.called for missing in self.branches)
        data = tuple(data.data_class for data in self.branches)

        self._model = model
        self.log_object = LogObject(prep, prep_attr, missing_data, data, results, options)
        self.entries = []

    def make_entry(self, comment: [str, None] = None) -> None:
        """
        Creates a new log entry by deep copying the current LogObject and adding an optional comment.

        Parameters
        ----------
        comment : str, optional
            A user-defined comment for the log entry.
        """
        new_log = deepcopy(self.log_object)
        new_log.add_comment(comment)
        self.entries.append(new_log)

    def set_model_from_log(self, entry_number: int) -> None:
        """
        Sets the model's state from the specified log entry.

        Parameters
        ----------
        entry_number : int
            The index of the log entry to be used for setting the model's state.

        Raises
        ------
        TypeError
            If the input entry_number is not an integer.
        """
        if not isinstance(entry_number, int):
            raise TypeError("Please input an int as entry_number")

        model_entry = deepcopy(self.entries[entry_number])

        self._model.results = model_entry.results
        self._model.options = model_entry.options
        self.branches = []
        if not self._model.single_branch:
            self._model.x = Branch(model_entry.data[0], self.branches)
            self._model.y = Branch(model_entry.data[1], self.branches)
            self.branches.append(self._model.x)
            self.branches.append(self._model.y)
        else:
            self._model.x = Branch(model_entry.data[0], self.branches)
            self.branches.append(self._model.x)

        [setattr(prep.preprocessing, "called", model_entry.prep[i]) for i, prep in enumerate(self.branches)]

        [setattr(prep.preprocessing, "scaling_attributes", model_entry.prep_attr[i])
         for i, prep in enumerate(self.branches)]

        [setattr(missing.missing_data, "called", model_entry.missing_data[i])
         for i, missing in enumerate(self.branches)]

        self.log_object = model_entry

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a summary of the log entries as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing a summary of the log entries.
        """
        summary = Summary(self.entries)
        return summary.return_dataframe()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Log instance.

        Returns
        -------
        str
            A string representation of the Log instance.
        """
        logs = self.entries.__repr__().replace('[', '').replace(']', '').replace(', ', '\n')
        if len(logs) == 0:
            logs = "None"

        return f"log entries:\n" \
               f"{logs}"


class Summary:
    """
    Can produce a summary of the log entries.

    Parameters
    ----------
    entries : list of LogObject
        A list of LogObject entries.

    Attributes
    ----------
    index : list of int
        The list of indices for the log entries.
    comment : list of str
        A list of comments associated with the log entries.
    date : list of date
        A list of dates when the log entries were created.
    time : list of time
        A list of times when the log entries were created.
    cv_type : list of str
        A list of cross-validation types for the log entries.
    cv_left_out : list of float
        A list of percentages of left-out data for cross-validation.
    opt_comp : list of int
        A list of optimal number of components for the log entries.
    x_prep : list of str
        A list of preprocessing functions applied to the X data.
    y_prep : list of str
        A list of preprocessing functions applied to the Y data.
    obs_removed : list of int
        A list of the number of removed observations for each log entry
    vars_removed : list of int
        A list of the number of removed variables for each log entry.
    rmsec : list of float
        A list of RMSE values for calibration data.
    rmsecv : list of float
        A list of RMSE values for cross-validation data.
    msec : list of float
        A list of MSE values for calibration data.
    msecv : list of float
        A list of MSE values for cross-validation data.
    biascv : list of float
        A list of bias values for cross-validation data.
    """
    def __init__(self, entries: list[LogObject, ...]):
        date_time = self.extract_value(entries, "created_at")
        self.index = [index for index in range(len(entries))]
        self.comment = self.extract_value(entries, "comment")
        self.date = [d.date() for d in date_time]
        self.time = [d.time() for d in date_time]
        cv_type = self.extract_value(entries, "_cross_validation", "options")
        called_functions = [cv.replace("_", " ") for cv in cv_type]
        self.cv_type = called_functions
        self.cv_left_out = self.extract_value(entries, "_percentage_left_out", "options")
        self.opt_comp = self.extract_value(entries, "optimal_number_component", "results")

        x_prep = self.extract_value(entries, "function", "x_prep")
        called_functions = [[prep.__name__.replace("_", " ") for prep in x] for x in x_prep]
        self.x_prep = [", ".join(called_function) for called_function in called_functions]

        y_prep = self.extract_value(entries, "function", "y_prep")
        called_functions = [[prep.__name__.replace("_", " ") for prep in y] for y in y_prep]
        self.y_prep = [", ".join(called_function) for called_function in called_functions]

        rows = self.extract_value(entries, "rows")
        obs_removed = [len(count_false(row)) for row in rows]
        self.obs_removed = obs_removed

        vars = self.extract_value(entries, "variables")
        vars_removed = [len(count_false(var)) for var in vars]
        self.vars_removed = vars_removed

        self.rmsec = self.extract_value_from_results(entries, "rmse", "calibration")
        self.rmsecv = self.extract_value_from_results(entries, "rmse", "cross_validation")
        self.msec = self.extract_value_from_results(entries, "mse", "calibration")
        self.msecv = self.extract_value_from_results(entries, "mse", "cross_validation")
        self.biascv = self.extract_value_from_results(entries, "bias", "cross_validation")

        self.return_dataframe()

    @staticmethod
    def extract_value(entries: list[LogObject, ...], key: str, inner: [None, str] = None) -> any:
        """
        Extracts the values of a given key from a list of LogObject instances.

        Parameters
        ----------
        entries : list[LogObject]
            A list of LogObject instances.
        key : str
            The key whose values need to be extracted from the LogObject instances.
        inner : str, optional
            An additional key to access an inner attribute, if needed.

        Returns
        -------
        list
            A list of values corresponding to the given key.
        """
        if not inner:
            result = [entry.__dict__.get(key) for entry in entries]
        else:
            result = [entry.__dict__.get(inner).__dict__.get(key) for entry in entries]
        return result

    def extract_value_from_results(self, entries: list[LogObject, ...], key: str, inner: [None, str] = None):
        """
        Extracts the values of a given key from the results of a list of LogObject instances.

        Parameters
        ----------
        entries : list[LogObject]
            A list of LogObject instances.
        key : str
            The key whose values need to be extracted from the results of the LogObject instances.
        inner : str, optional
            An additional key to access an inner attribute, if needed.

        Returns
        -------
        list
            A list of values corresponding to the given key.
        """
        result = [
            entry.results.__dict__.get(inner).__dict__.get(key)
            if entry.results.__dict__.get(inner) is not None else "Not calculated"
            for entry in entries
        ]

        results = [
            res[self.opt_comp[i]]
            if self.opt_comp[i] is not None else "Not calculated"
            for i, res in enumerate(result)
        ]

        return results

    def return_dataframe(self) -> pd.DataFrame:
        """
        Returns a summary as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the summary information of the log entries.
        """
        df = pd.DataFrame().from_dict(self.__dict__)
        df.columns = df.columns.str.replace("_", " ")
        return df


def count_false(boolean: list[bool]) -> tuple:
    return tuple(filter(lambda i: not boolean[i], range(len(boolean))))
