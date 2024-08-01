# standard library imports
import logging
import logging.handlers
import multiprocessing
import sys
from typing import Optional, Union

# 3rd party library imports

# local imports


class LogHandlerBase:
    def __init__(self):
        self.__init_logger()

    def __init_logger(self):
        self._my_logger = logging.getLogger(name=f"{__name__}-{self.__class__.__name__}")

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter()
        handler.setFormatter(formatter)
        self._my_logger.addHandler(handler)


class CustomFormatter(logging.Formatter):
    """
    Helps to format terminal color and organize information in a log message.
    based on: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    grey = "\x1b[37;20m"
    cyan = "\x1b[36;20m"
    violet = "\x1b[35;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    f = '[%(levelname)s] [%(name)s] (%(asctime)s) |%(name)s|: %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {
        logging.DEBUG: grey + f + reset,
        logging.INFO: green + f + reset,
        logging.WARNING: yellow + f + reset,
        logging.ERROR: red + f + reset,
        logging.CRITICAL: bold_red + f + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class StandardLogger:
    """
    Base class for capturing event information (indicating that something happened) and storing it appropriately.
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        Logger.setup_file_handler(
            logger=self.__class__.__name__,
            handler_level=logging.INFO
        )
        Logger.setup_stdout_handler(
            logger=self.__class__.__name__,
            handler_level=logging.DEBUG
        )

        self._logger.info(f"LOGGER SET UP: {self.__class__.__name__}")


class Logger:
    log_filename = "logs.log"
    propagate = True  # propagate all messages from loggers to main logger

    @staticmethod
    def cleanup_logger_handlers(
            logger: Optional[Union[str, logging.Logger]] = None
    ) -> None:
        """
        Cleans up all handlers of particular logger.

        :param logger: chosen logger or its name.
        :return: None.
        """

        if type(logger) == str:
            current_logger = logging.getLogger(logger)
        elif type(logger) == logging.Logger:
            current_logger = logger
        else:
            raise ValueError(f"Argument `logger` should be of type String or logging.Logger or None. "
                             f"Current Type: {type(logger)}")

        current_logger.handlers = []

    @staticmethod
    def change_logger_level(
            level,
            logger: Optional[Union[str, logging.Logger]] = None
    ) -> None:
        """
        Changes logging level of particular logger.

        :param level: new threshold level of logger. All messages that are at least level-important will be proceeded by
            this logger.
        :param logger: chosen logger or its name.
        :return: None.
        """

        if type(logger) == str:
            current_logger = logging.getLogger(logger)
        elif type(logger) == logging.Logger:
            current_logger = logger
        else:
            raise ValueError(f"Argument `logger` should be of type String or logging.Logger or None. "
                             f"Current Type: {type(logger)}")

        current_logger.setLevel(level=level)

    @staticmethod
    def setup_file_handler(
            handler_level=logging.NOTSET,
            logger=None
    ) -> None:
        """
        Adds TimedRotatingFileHandler to particular logger.

        :param handler_level: threshold level of handler. All messages that are at least level-important will be
            proceeded by this handler.
        :param logger: chosen logger or its name.
        :return: None.
        """

        if type(logger) == str:
            current_logger = logging.getLogger(logger)
        elif type(logger) == logging.Logger:
            current_logger = logger
        else:
            raise ValueError(f"Argument `logger` should be of type String or logging.Logger or None. "
                             f"Current Type: {type(logger)}")

        current_logger.propagate = Logger.propagate

        rotating_handler = logging.handlers.TimedRotatingFileHandler(filename=Logger.log_filename, when='d', interval=1)
        formatter = logging.Formatter('[%(levelname)s | %(name)s]:%(asctime)s -%(module)s- %(message)s')
        rotating_handler.setFormatter(formatter)
        rotating_handler.setLevel(level=handler_level)
        current_logger.addHandler(rotating_handler)

    @staticmethod
    def setup_stdout_handler(
            handler_level=logging.NOTSET,
            logger: Optional[Union[str, logging.Logger]] = None
    ) -> None:
        """
        Adds stdout handler to particular logger.

        :param handler_level: threshold level of handler. All messages that are at least level-important will be
            proceeded by this handler.
        :param logger: chosen logger or its name.
        :return: None.
        """

        if type(logger) == str:
            current_logger = logging.getLogger(logger)
        elif type(logger) == logging.Logger:
            current_logger = logger
        else:
            raise ValueError(f"Argument `logger` should be of type String or logging.Logger or None. "
                             f"Current Type: {type(logger)}")

        current_logger.propagate = Logger.propagate

        streaming_handler = logging.StreamHandler()
        streaming_handler.setFormatter(CustomFormatter())
        streaming_handler.setLevel(level=handler_level)
        current_logger.addHandler(streaming_handler)

    @staticmethod
    def setup_queue_handler(
            queue: multiprocessing.Queue,
            handler_level=logging.NOTSET,
            logger: Optional[Union[str, logging.Logger]] = None
    ) -> None:
        """
        Adds queue handler to particular logger.

        :param queue: TODO
        :param handler_level: threshold level of handler. All messages that are at least level-important will be
            proceeded by this handler.
        :param logger: chosen logger or its name.
        :return: None.
        """

        if type(logger) == str:
            current_logger = logging.getLogger(logger)
        elif type(logger) == logging.Logger:
            current_logger = logger
        else:
            raise ValueError(f"Argument `logger` should be of type String or logging.Logger or None. "
                             f"Current Type: {type(logger)}")

        current_logger.propagate = Logger.propagate

        queue_handler = logging.handlers.QueueHandler(queue=queue)
        queue_handler.setLevel(level=handler_level)
        current_logger.addHandler(queue_handler)



