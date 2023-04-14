import datetime
from copy import deepcopy

from me3cs.framework.helper_classes.options import Options

from me3cs.framework.branch import Branch
from me3cs.framework.results import Results
from me3cs.preprocessing.called import Called


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

    def __repr__(self) -> str:
        return f"model type: {self.last_model_called} created {self.created_at}"

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


class Log:
    def __init__(self, branches: [list[Branch, Branch], list[Branch]], results: Results, options: Options):

        prep = tuple(prep.preprocessing.called for prep in branches)
        missing_data = tuple(missing.missing_data.called for missing in branches)

        rows = branches[0]._row_index
        self.log_object = LogObject(prep, missing_data, results, options, rows)
        self.entries = []

    def make_entry(self):
        new_log = deepcopy(self.log_object)

        self.entries.append(new_log)

    def __repr__(self):
        logs = self.entries.__repr__().replace('[', '').replace(']', '').replace(', ', '\n')
        if len(logs) == 0:
            logs = "None"

        return f"log entries:\n" \
               f"{logs}"

    def get_summary(self) -> dict:

        pass


def extract_values(lst, *keys):
    """
    Given a list of dictionaries, extract the values corresponding to the given keys.

    Args:
        lst (list[dict]): A list of dictionaries.
        keys (str): The keys to extract values for.

    Returns:
        list: A list of tuples, where each tuple contains the values corresponding
        to the given keys in each dictionary.
    """
    return [tuple(d.get(key) for key in keys) for d in lst]
