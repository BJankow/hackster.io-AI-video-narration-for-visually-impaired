# standard library imports

# 3rd party library imports

# local imports


class lazyproperty:
    """
    A property that evaluates given function only ones.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)  # passing instance of other class to self

            # Change value of an attribute (which key is name of decorated function) in instance of other class from
            # function reference to value this function returns. This way computation happens only ones (only ones the
            # function is called). Next calling this name will not call __get__ but __dict__ as it possesses searched
            # key
            setattr(instance, self.func.__name__, value)
        return value
