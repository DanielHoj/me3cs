import datetime
import typing
from copy import deepcopy

import pandas as pd

from me3cs.framework.branch import Branch
from me3cs.framework.data import Data
from me3cs.framework.helper_classes.options import Options
from me3cs.framework.outlier_detection import count_false
from me3cs.framework.results import Results
from me3cs.preprocessing.called import Called

if typing.TYPE_CHECKING:
    from me3cs.framework.base_model import BaseModel


class LogObject:
    def __init__(self,
                 prep: tuple[Called, ...],
                 missing_data: tuple[Called, ...],
                 data: tuple[Data, ...],
                 results: Results,
                 options: Options,
                 ) -> None:

        if len(prep) > 0:
            self.x_prep, self.y_prep = prep
            self.x_missing, self.y_missing = missing_data
            self.x_data, self.y_data = data
        else:
            self.x_prep = prep[0]
            self.x_missing = missing_data[0]
            self.x_data = data[0]

        self.results = results
        self.options = options

        self.prep = prep
        self.missing_data = missing_data
        self.data = data
        self.rows = data[0].rows.total
        self.variables = data[0].variables.total

        self.created_at = datetime.datetime.now().replace(microsecond=0)
        self.last_model_called = None
        self.comment = None

    def __repr__(self) -> str:
        return f"model type: {self.last_model_called} - created {self.created_at}"

    def __copy__(self):
        new_obj = LogObject(self.prep, self.missing_data, self.data, self.results, self.options)
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def __deepcopy__(self, memo):
        new_obj = LogObject(deepcopy(self.prep, memo), deepcopy(self.missing_data, memo),
                            deepcopy(self.data, memo), deepcopy(self.results, memo),
                            deepcopy(self.options, memo),
                            )
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def add_comment(self, comment: [None, str]):
        if not (isinstance(comment, str) or comment is None):
            raise TypeError("comment needs to be a string")
        self.comment = comment


class Log:
    def __init__(self, model: "BaseModel", results: Results,
                 options: Options):

        self.branches = model.branches
        prep = tuple(prep.preprocessing.called for prep in self.branches)
        missing_data = tuple(missing.missing_data.called for missing in self.branches)
        data = tuple(data.data_class for data in self.branches)

        self._model = model
        self.log_object = LogObject(prep, missing_data, data, results, options)
        self.entries = []

    def make_entry(self, comment: [str, None] = None) -> None:
        new_log = deepcopy(self.log_object)
        new_log.add_comment(comment)
        self.entries.append(new_log)

    def set_model_from_log(self, entry_number: int) -> None:

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
        [setattr(missing.missing_data, "called", model_entry.missing_data[i]) for i, missing in enumerate(self.branches)]

        self.log_object = model_entry

    def __repr__(self):
        logs = self.entries.__repr__().replace('[', '').replace(']', '').replace(', ', '\n')
        if len(logs) == 0:
            logs = "None"

        return f"log entries:\n" \
               f"{logs}"

    def get_summary(self) -> pd.DataFrame:
        summary = Summary(self.entries)
        return summary.return_dataframe()


class Summary:
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
    def extract_value(entries: list[LogObject, ...], key: str, inner: [None, str] = None):
        if not inner:
            result = [entry.__dict__.get(key) for entry in entries]
        else:
            result = [entry.__dict__.get(inner).__dict__.get(key) for entry in entries]
        return result

    def extract_value_from_results(self, entries: list[LogObject, ...], key: str, inner: [None, str] = None):

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
        df = pd.DataFrame().from_dict(self.__dict__)
        df.columns = df.columns.str.replace("_", " ")
        return df
