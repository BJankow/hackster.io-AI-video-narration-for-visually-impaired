# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


# 3rd party library imports

# local imports

class StagesProcessorInterface(ABC):

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def load_movie(self, path: Union[str, Path]):
        """
        Given path to movie file loads it to object's memory as attribute.

        :param path: path to movie being loaded.
        :return:
        """
        pass

    @abstractmethod
    def detect_cuts(self, *args, **kwargs):
        """
        Detects cuts in movie - logical splitting movie into shots/scenes.

        :return:
        """
        pass

    @abstractmethod
    def generate_descriptions(self):
        """
        Generates descriptions for every shot/scene.

        :return:
        """
        pass

    @abstractmethod
    def synthesize_descriptions(self):
        """
        Synthesizes all text shot/scene descriptions into voice sound.

        :return:
        """
        pass

    @abstractmethod
    def compose_movie(self, out_fp: Union[str, Path]):
        """
        Composes frames and sound into movie.

        :param out_fp: path to file where the composed movie will be saved.
        :return:
        """
        pass

