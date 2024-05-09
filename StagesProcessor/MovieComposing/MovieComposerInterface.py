# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

# 3rd party library imports

# local imports


class MovieComposerInterface(ABC):

    @abstractmethod
    def compose(self, video, audio):
        """
        Composes given frames and sound into movie.

        :param video: video for movie.
        :param audio: audio for movie.
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
