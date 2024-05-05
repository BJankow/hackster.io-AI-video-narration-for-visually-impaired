# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

# 3rd party library imports

# local imports


class MovieComposerInterface(ABC):

    @abstractmethod
    def compose(self, frames: List, sound):
        """
        Composes given frames and sound into movie.

        :param frames: sequence of frames for movie.
        :param sound: sound for movie.
        :return:
        """
        pass

    @abstractmethod
    def save(self, out_fp: Union[str, Path]):
        """
        Saves composed movie as a file.

        :param out_fp: path to output file.
        :return:
        """
        pass
