import numpy as np

from me3cs.misc.handle_data import handle_zeros_in_scale
from me3cs.misc.preprocessing import preprocessing_scaling
from me3cs.preprocessing.base import PreprocessingBaseClass
from me3cs.preprocessing.called import set_called


def scale_once(func):
    def inner(self, *args, **kwargs):
        if not self.data_is_centered:
            func(self, *args, **kwargs)
        else:
            functions_names = [function.__qualname__ for function in self.called.function]
            scaling_elements = [i for i in functions_names if i.startswith('Scaling')]
            if scaling_elements:
                removed_element = scaling_elements[0]
                index = functions_names.index(removed_element)

                self.called.function.pop(index)
                self.called.args.pop(index)
                self.called.kwargs.pop(index)

                func(self, *args, **kwargs)
                self.data = self._missing_data_link.get()

                for function, args, kwargs in zip(
                        self.called.function, self.called.args, self.called.kwargs
                ):
                    function(self, *args, **kwargs)

    return inner


class Scaling(PreprocessingBaseClass):
    def __scale_pipeline(self, constant: [np.ndarray | float], scale: [np.ndarray | float]) -> None:
        data = self.data

        new = preprocessing_scaling(data, constant, scale)
        self.data = new
        self.update_is_centered(True)

    @scale_once
    @set_called
    def autoscale(self) -> None:
        self.set_ref()

        constant = -self.reference.mean
        scale = handle_zeros_in_scale(self.reference.std)
        self.__scale_pipeline(constant, scale)

    @scale_once
    @set_called
    def mean_center(self) -> None:
        self.set_ref()

        constant = -self.reference.mean
        scale = 1.0
        self.__scale_pipeline(constant, scale)

    @scale_once
    @set_called
    def pareto(self) -> None:
        self.set_ref()

        constant = -self.reference.mean
        scale = handle_zeros_in_scale(self.reference.sqrt_std)

        self.__scale_pipeline(constant, scale)

    @scale_once
    @set_called
    def median_center(self) -> None:
        self.set_ref()

        constant = -self.reference.median
        scale = 1.0

        self.__scale_pipeline(constant, scale)
