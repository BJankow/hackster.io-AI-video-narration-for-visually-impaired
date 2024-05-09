# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, List


# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip

# local imports


class StagesProcessorInterface(ABC):

    def __init__(
            self
    ):
        pass

    @abstractmethod
    def load_movie(self, fp: Union[str, Path]) -> Tuple:
        """
        Given path to movie file loads it to object's memory as attribute.

        :param fp: path to movie file.
        :return:
        """
        pass

    @abstractmethod
    def detect_scenes(self, fp: Union[str, Path], *args, **kwargs) -> List:
        """
        Detects scenes in movie.

        :param fp: path to movie file.
        :return:
        """
        pass

    @abstractmethod
    def generate_descriptions(self, fp: Union[str, Path], scenes: List) -> List[str]:
        """
        Generates descriptions for every shot/scene.

        :param fp: path to movie file.
        :param scenes: scenes of given movie.
        :return: List of text descriptions.
        """
        pass

    @abstractmethod
    def synthesize_descriptions(self, fp: Union[str, Path], descriptions: List[str]) -> List:
        """
        Synthesizes all text shot/scene descriptions into voice sound.

        :param fp: path to movie file.
        :param descriptions: List of text descriptions.
        :return: List of synthesized descriptions.
        """
        pass

    @abstractmethod
    def compose_movie(
            self,
            fp: Union[str, Path],
            video: VideoFileClip,
            audio: AudioFileClip,
            out_fp: Union[str, Path]
    ):
        """
        Composes frames and sound into movie.

        :param fp: path to original movie file.
        :param video: video of new movie.
        :param audio: audio of new movie.
        :param out_fp: path to file where the composed movie will be saved.
        :return:
        """
        pass

