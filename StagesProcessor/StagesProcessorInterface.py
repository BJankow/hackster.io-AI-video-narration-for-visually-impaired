# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from pydub.audio_segment import AudioSegment
from typing import Union, Tuple, List


# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from scenedetect import FrameTimecode


# local imports


class StagesProcessorInterface(ABC):

    @abstractmethod
    def load_movie(self, fp: Union[str, Path]) -> Tuple:
        """
        Given path to movie file loads it to object's memory as attribute.

        :param fp: path to movie file.
        :return:
        """

    @abstractmethod
    def detect_scenes(self, fp: Union[str, Path], *args, **kwargs) -> List:
        """
        Detects scenes in movie.

        :param fp: path to movie file.
        :return:
        """

    @abstractmethod
    def generate_descriptions(self, fp: Union[str, Path], scenes: List) -> List[str]:
        """
        Generates descriptions for every shot/scene.

        :param fp: path to movie file.
        :param scenes: scenes of given movie.
        :return: List of text descriptions.
        """

    @abstractmethod
    def synthesize_descriptions(self, fp: Union[str, Path], descriptions: List[str]) -> List:
        """
        Synthesizes all text shot/scene descriptions into voice sound.

        :param fp: path to movie file.
        :param descriptions: List of text descriptions.
        :return: List of synthesized descriptions.
        """

    @abstractmethod
    def compose_movie(
            self,
            fp: Union[str, Path],
            out_fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
            synthesized_descriptions: List[AudioSegment]
    ):
        """
        Composes frames and sound with synthesized descriptions into movie.

        :param fp: path to original movie file.
        :param out_fp: path to file where the composed movie will be saved.
        :param scenes: scenes as Tuple of FrameTimecodes. First indicates beginning of the scene, second - end.
        :param synthesized_descriptions: descriptions as audio.
        :return:
        """

