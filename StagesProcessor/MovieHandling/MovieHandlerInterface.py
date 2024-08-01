# standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

# 3rd party library imports
from pydub import AudioSegment
from scenedetect import VideoStream

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

    @abstractmethod
    def get_video(self) -> Optional[VideoStream]:
        """
        Obtains video from previously loaded movie.

        :return: video
        """
        pass

    @abstractmethod
    def get_audio(self) -> Optional[AudioSegment]:
        """
        Obtains audio from previously loaded movie.

        :return: audio
        """
        pass
