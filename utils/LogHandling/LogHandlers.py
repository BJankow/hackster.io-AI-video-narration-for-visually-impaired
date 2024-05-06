# standard library imports
import logging
import sys

# 3rd party library imports

# local imports


class LogHandlerBase:
    def __init__(self):
        self.__init_logger()

    def __init_logger(self):
        self._my_logger = logging.getLogger(name=f"{__name__}-{self.__class__.__name__}")

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(levelname)s] [%(name)s] (%(asctime)s) |%(name)s|: %(message)s')
        handler.setFormatter(formatter)
        self._my_logger.addHandler(handler)  # TODO: check if it works to stdout
