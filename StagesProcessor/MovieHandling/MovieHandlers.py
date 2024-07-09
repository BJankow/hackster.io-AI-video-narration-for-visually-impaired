# standard library imports
import os.path
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Tuple

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from scenedetect import VideoStreamCv2
import soundfile as sf

# local imports
from .MovieHandlerInterface import MovieHandlerInterface
from utils.LogHandling.LogHandlers import StandardLogger


class MovieHandlerBase(MovieHandlerInterface, StandardLogger):

    def __init__(self):
        super(MovieHandlerInterface, self).__init__()
        self._video: Optional[VideoStreamCv2] = None
        self._audio: Optional[AudioSegment] = None

    def load(self, fp: Union[str, Path]) -> Tuple[VideoStreamCv2, AudioSegment]:
        self._video = VideoStreamCv2(fp)
        self._audio = AudioSegment.from_file(fp)

        return self._video, self._audio

    def get_video(self) -> Optional[VideoStreamCv2]:
        return self._video

    def get_audio(self) -> Optional[AudioFileClip]:
        return self._audio

