# standard library imports
import os.path
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from scenedetect import VideoStream
import soundfile as sf

# local imports
from .MovieHandlerInterface import MovieHandlerInterface


class MovieHandlerBase(MovieHandlerInterface):

    def __init__(self):
        super(MovieHandlerInterface, self).__init__()
        self.__video: Optional[VideoStream] = None
        self.__audio: Optional[AudioFileClip] = None

    def load(self, fp: Union[str, Path]):
        if self.__video is None:
            self.__video = VideoFileClip(fp)

    def get_video(self) -> Optional[VideoStream]:
        return self.__video

