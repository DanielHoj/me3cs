import datetime
from copy import deepcopy

from me3cs.framework.branch import Branch
from me3cs.framework.results import Results


class LogObject:
    def __init__(self, branches: [list[Branch, Branch], list[Branch]], results: Results):
        self._branches = branches
        if len(branches) > 0:
            x, y = branches
            self.x_prep = x.preprocessing.called
            self.y_prep = y.preprocessing.called
            # self.x_missing = x.missing_data.called
            # self.y_missing = y.missing_data.called
        else:
            x = branches
            self.x_prep = x.preprocessing.called
            # self.x_missing = x.missing_data.called
        self.results = results
        self.rows = x._row_index.get_total_index()
        self.created_at = datetime.datetime.now().replace(microsecond=0)
        self.last_model_called = None

    def __repr__(self):
        return f"model type: {self.last_model_called} created {self.created_at}"

    def __copy__(self):
        new_obj = LogObject(self._branches, self.results)
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj

    def __deepcopy__(self, memo):
        new_obj = LogObject(deepcopy(self._branches, memo), deepcopy(self.results, memo))
        new_obj.created_at = datetime.datetime.now().replace(microsecond=0)
        new_obj.last_model_called = self.last_model_called
        return new_obj


class Log:
    def __init__(self, branches: [list[Branch, Branch], list[Branch]], results: Results):
        self._x_branch = branches[0]
        if len(branches) > 0:
            self._y_branch = branches[0]
        self._results = results
        self.model_details = LogObject(branches, results)
        self.entries = []

    def make_entry(self):
        new_log = deepcopy(self.model_details)

        self.entries.append(new_log)

    def get_summary(self):
        pass

    def __repr__(self):
        logs = self.entries.__repr__().replace('[', '').replace(']', '').replace(', ', '\n')
        if len(logs) == 0:
            logs = "None"

        return f"me3cs Models created:\n" \
               f"{logs}"
