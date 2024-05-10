# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Tuple

# 3rd party library imports
from pydub.audio_segment import AudioSegment
from scenedetect import FrameTimecode


# local imports


class MovieComposerInterface(ABC):

    @abstractmethod
    def compose(
            self,
            video_fp: Union[str, Path],
            audio_fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
            synthesized_descriptions: List[AudioSegment]
    ):
        """
        Composes Video and Audio files into movie.
        :param video_fp: filepath to video file.
        :param audio_fp: filepath to audio file.
        :param scenes: scenes as Tuple of FrameTimecodes. First indicates beginning of the scene, second - end.
        :param synthesized_descriptions: descriptions as audio.
        :return:
        """

    @abstractmethod
    def save(self, out_fp: Union[str, Path]):
        """
        Saves composed movie as a file.

        :param out_fp: path to output file.
        :return:
        """
