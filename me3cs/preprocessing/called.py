def set_called(func):

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
    def __init__(self, function, args, kwargs):
        self.function: list = function
        self.args: list = args
        self.kwargs: list = kwargs
