import numpy as np

from me3cs.misc.handle_data import handle_zeros_in_scale
from me3cs.misc.preprocessing import preprocessing_scaling
from me3cs.preprocessing.base import PreprocessingBaseClass
from me3cs.preprocessing.called import set_called


def scale_once(func):
    """Decorator function to ensure scaling functions are only called once.

    Parameters:
    -----------
    func : function
        A preprocessing method to be decorated.

    Returns:
    --------
    inner : function
        The decorated function.

    Notes:
    ------
    This decorator function checks if the data has already been centered before calling the decorated function. If
    the data has not been centered, the decorated function is called as usual. If the data has been previously
    centered, it removes the previous scaling function from the call history and re-applies the function with the new
    parameters. This is to ensure that scaling functions are only applied once to the data.
    """
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
                self.data = self.data_class.get_raw_data()

                for function, args, kwargs in zip(
                        self.called.function, self.called.args, self.called.kwargs
                ):
                    function(self, *args, **kwargs)

    return inner


class Scaling(PreprocessingBaseClass):
    """
    A class containing scaling methods for preprocessing data.

    Methods
    -------
    autoscale()
        Scale the data to have zero mean and unit variance.
    mean_center()
        Subtract the mean from the data.
    pareto()
        Scale the data using square root of standard deviation.
    median_center()
        Subtract the median from the data.
    """

    def _scale_pipeline(self, constant: [np.ndarray | float], scale: [np.ndarray | float]) -> None:
        """
        Perform scaling of data using specified constant and scale. updates the is_centered instance variable to True.

        Parameters
        ----------
        constant : numpy.ndarray or float
            Constant value used for scaling.
        scale : numpy.ndarray or float
            Scaling factor.

        """
        data = self.data

        new = preprocessing_scaling(data, constant, scale)
        self.data = new
        self.update_is_centered(True)

    @scale_once
    @set_called
    def autoscale(self) -> None:
        """
        Scale the data to have zero mean and unit variance.
        """

        match self.mode:
            case "preprocess":
                constant = self.data.mean(axis=0)
                self.scaling_attributes.mean = constant

                scale = handle_zeros_in_scale(self.data.std(axis=0))
                self.scaling_attributes.std = scale
                self._scale_pipeline(-constant, scale)

            case "reference":
                constant = self._reference.mean(axis=0)
                scale = handle_zeros_in_scale(self._reference.std(axis=0))
                self._scale_pipeline(-constant, scale)

            case "predict":
                try:
                    constant = self.scaling_attributes.mean
                    scale = self.scaling_attributes.std
                    self._scale_pipeline(-constant, scale)
                except ValueError:
                    print("Autoscaling has not been called")

    @scale_once
    @set_called
    def mean_center(self) -> None:
        """
        Subtract the mean from the data.
        """
        match self.mode:
            case "preprocess":
                constant = self.data.mean(axis=0)
                self.scaling_attributes.mean = constant
                self._scale_pipeline(-constant, 1.0)

            case "reference":
                constant = self._reference.mean(axis=0)
                self._scale_pipeline(-constant, 1.0)

            case "predict":
                try:
                    constant = self.scaling_attributes.mean
                    self._scale_pipeline(-constant, 1.0)
                except ValueError:
                    print("mean_center has not been called")

    @scale_once
    @set_called
    def pareto(self) -> None:
        """
        Scale the data using square root of standard deviation, and subtracts the mean.
        """
        match self.mode:
            case "preprocess":
                constant = self.data.mean(axis=0)
                self.scaling_attributes.mean = constant

                scale = handle_zeros_in_scale(np.sqrt(self.data.std(axis=0)))
                self.scaling_attributes.sqrt_std = scale

                self._scale_pipeline(-constant, scale)

            case "reference":
                constant = self._reference.mean(axis=0)
                scale = handle_zeros_in_scale(np.sqrt(self._reference.std(axis=0)))
                self._scale_pipeline(-constant, scale)

            case "predict":
                try:
                    constant = self.scaling_attributes.mean
                    scale = self.scaling_attributes.sqrt_std
                    self._scale_pipeline(-constant, scale)

                except ValueError:
                    print("Autoscaling has not been called")

    @scale_once
    @set_called
    def median_center(self) -> None:
        """
        Subtract the median from the data.
        """
        match self.mode:
            case "preprocess":
                constant = np.median(self.data, axis=0)
                self.scaling_attributes.median = constant
                self._scale_pipeline(-constant, 1.0)

            case "reference":
                constant = np.median(self._reference, axis=0)
                self._scale_pipeline(-constant, 1.0)

            case "predict":
                try:
                    constant = self.scaling_attributes.median
                    self._scale_pipeline(-constant, 1.0)
                except ValueError:
                    print("mean_center has not been called")
