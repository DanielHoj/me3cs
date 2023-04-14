import datetime
from copy import deepcopy
import typing

from me3cs.framework.helper_classes.options import Options
from me3cs.framework.branch import Branch
from me3cs.framework.results import Results
from me3cs.preprocessing.called import Called

if typing.TYPE_CHECKING:
    from me3cs.framework.base_model import BaseModel


class LogObject:
    def __init__(self,
                 prep: tuple[Called, ...],
                 missing_data: tuple[Called, ...],
                 results: Results,
                 options: Options,
                 rows: list[bool, ...],
                 ) -> None:

        if len(prep) > 0:
            self.x_prep = prep[0]
            self.y_prep = prep[1]
            self.x_missing = missing_data[0]
            self.y_missing = missing_data[1]
        else:
            self.x_prep = prep[0]
            self.x_missing = missing_data[0]

        self.results = results
        self.options = options

        self._prep = prep
        self._missing_data = missing_data

        self.rows = rows
        self.created_at = datetime.datetime.now().replace(microsecond=0)
        self.last_model_called = None
        self.comment = None

    def __repr__(self) -> str:
        return f"model type: {self.last_model_called} - created {self.created_at}"

    def __copy__(self):
        new_obj = LogObject(self._prep, self._missing_data, self.results, self.options, self.rows)
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def __deepcopy__(self, memo):
        new_obj = LogObject(deepcopy(self._prep, memo), deepcopy(self._missing_data, memo),
                            deepcopy(self.results, memo), deepcopy(self.options, memo),
                            deepcopy(self.rows, memo))
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def add_comment(self, comment: [None, str]):
        if not (isinstance(comment, str) or comment is None):
            raise TypeError("comment needs to be a string")
        self.comment = comment


class Log:
    def __init__(self, model: "BaseModel", branches: [list[Branch, Branch], list[Branch]], results: Results, options: Options):

        prep = tuple(prep.preprocessing.called for prep in branches)
        missing_data = tuple(missing.missing_data.called for missing in branches)
        rows = branches[0]._row_index

        self._model = model
        self.log_object = LogObject(prep, missing_data, results, options, rows)
        self.entries = []

    def make_entry(self, comment: [str, None] = None) -> None:
        new_log = deepcopy(self.log_object)
        new_log.add_comment(comment)
        self.entries.append(new_log)

    def set_model_from_log(self, entry_number: int) -> None:

        if not isinstance(entry_number, int):
            raise TypeError("Please input an int as entry_number")

        model_entry = self.entries[entry_number]

        self._model.results = model_entry.results
        self._model.options = model_entry.options

        if not self._model._single_branch:
            # set called
            self._model.x.preprocessing.called = model_entry.x_prep
            self._model.x.missing_data.called = model_entry.x_missing

            self._model.y.preprocessing.called = model_entry.y_prep
            self._model.y.missing_data.called = model_entry.y_missing

            self._model.x._row_index = model_entry.rows
            self._model.y._row_index = model_entry.rows

            # call in order
            self._model.x._reset_link()
            self._model.x.missing_data.call_in_order()
            self._model.x.preprocessing.call_in_order()

            self._model.y.missing_data.call_in_order()
            self._model.y.preprocessing.call_in_order()

        else:
            # Set called
            self._model.x.preprocessing.called = model_entry.x_prep
            self._model.x.missing_data.called = model_entry.x_missing
            self._model.x._row_index = model_entry.rows

            # Call in order
            self._model.x._reset_link()
            self._model.x.missing_data.call_in_order()
            self._model.x.preprocessing.call_in_order()

        self._model.x._update_data_from_index()

    def __repr__(self):
        logs = self.entries.__repr__().replace('[', '').replace(']', '').replace(', ', '\n')
        if len(logs) == 0:
            logs = "None"

        return f"log entries:\n" \
               f"{logs}"

    def get_summary(self) -> dict:

        pass
