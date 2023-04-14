def set_called(func):
    """
    Decorator that adds the decorated function to the `self.called` list of the object,
    along with any arguments or keyword arguments provided.
    This is used to track which methods of the 'preporcessing' module that has been called

    Parameters:
    -----------
    func : function
        The function being decorated.

    Returns:
    --------
    function
        A wrapped function that calls `func` and updates the `self.called` list with information about the
        function call.
    """
    def inner(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.called.function.append(func)
        if args:
            self.called.args.append(args)
        else:
            self.called.args.append(())
        if kwargs:
            self.called.kwargs.append(kwargs)
        else:
            self.called.kwargs.append({})

    return inner


class Called:
    """
    A class to represent the methods that have been called in a module, either the
    preprocessing or missing data module.

    Attributes
    ----------
    function : list
        The list of function objects that have been called.
    args : list
        The list of arguments passed to the called functions.
    kwargs : list
        The list of keyword arguments passed to the called functions.
    """
    def __init__(self, function, args, kwargs):
        self.function: list = function
        self.args: list = args
        self.kwargs: list = kwargs

    def reset(self):
        self.function.clear()
        self.args.clear()
        self.kwargs.clear()

    def __repr__(self):
        called_functions = [x.__name__ for x in self.function]
        if len(called_functions) == 0:
            called_functions = ["None"]
        called_functions_as_str = ", ".join(called_functions)
        return f"Called functions: {called_functions_as_str}"

