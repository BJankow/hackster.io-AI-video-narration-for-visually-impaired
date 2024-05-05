# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

# 3rd party library imports

# local imports


class MovieHandlerInterface(ABC):

    @abstractmethod
    def load(self, fp: Union[str, Path]):
        """
        Loads movie into object's memory as attribute.

        :param fp: path to movie file.
        :return: None.
        """
        pass

