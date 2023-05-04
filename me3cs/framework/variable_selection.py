from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_model import BaseModel


class VariableSelection:
    def __init__(self, model: "BaseModel"):
        self._model = model

    def remove_variables(self, outlier_index: [tuple[..., int], int]):
        self._model.x.data_class.remove_columns("outlier_detection", outlier_index)
        self._model.x.preprocessing.call_in_order()
        call_model(self)

    def range_cut(self, range_min: int, range_max: int):
        if not (isinstance(range_min, int) and isinstance(range_max, int)):
            raise TypeError("Inputs need to be ints")

        outlier_index = tuple(range(range_min, range_max))
        self._model.x.data_class.remove_columns("outlier_detection", outlier_index)
        self._model.x.preprocessing.call_in_order()
        call_model(self)

    def range_keep(self, range_min: int, range_max: int):
        if not (isinstance(range_min, int) and isinstance(range_max, int)):
            raise TypeError("Inputs need to be ints")

        variables = tuple(i for i in range(self._model.x.data.shape[1]))
        outlier_index = tuple(val for val in variables if val < range_min or val > range_max)
        self._model.x.data_class.remove_columns("outlier_detection", outlier_index)
        self._model.x.preprocessing.call_in_order()
        call_model(self)

    def reset(self):
        self._model.x.data_class.reset_index("outlier_detection")
        self._model.x.preprocessing.call_in_order()


def call_model(self):
    if self._model.log.log_object.last_model_called:
        mdl_type = self._model.log.log_object.last_model_called.lower()
        model = getattr(self._model, mdl_type)
        model()
