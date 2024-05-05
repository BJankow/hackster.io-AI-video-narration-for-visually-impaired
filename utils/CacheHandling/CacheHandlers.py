# standard library imports
from joblib import Memory

# 3rd party library imports


# local imports


class CacheHandlerBase:
    def __init__(
            self,
            **kwargs
    ):
        try:
            self.__init_cache_handler(
                cache_location=kwargs.pop("cache_location"),
                cache_backend=kwargs.pop("cache_backend"),
                verbose=kwargs.pop("verbose")
            )
        except KeyError as e:
            raise KeyError(f"Argument required: {e}")
        except Exception as e:
            raise Exception(f"UNHANDLED EXCEPTION: {e}")

    def __init_cache_handler(
            self,
            cache_location: str,
            cache_backend: str = 'local',
            verbose: bool = True
    ):
        """
        Initializes cache handler for this class

        :param cache_location: The path of the base directory to use as a data store.
        :param cache_backend: Type of store backend for reading/writing cache files.
        :param verbose: Verbosity flag, controls the debug messages that are issued as functions are evaluated.
        :return:
        """
        self.__memory = Memory(
            location=cache_location,
            backend=cache_backend,
            verbose=verbose
        )
