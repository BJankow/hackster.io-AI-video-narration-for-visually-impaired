# standard library imports
from pathlib import Path
from typing import List, Union

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip

# local imports
from .MovieComposerInterface import MovieComposerInterface
from utils.LogHandling.LogHandlers import StandardLogger


class MovieComposerBase(MovieComposerInterface, StandardLogger):

    def __init__(self):
        super(MovieComposerBase, self).__init__()
        self.__movie = None

    def compose(self, video: VideoFileClip, audio: AudioFileClip):
        self.__movie = video.set_audio(audio)

    def save(self, out_fp: Union[str, Path]):
        self.__movie.write_videofile(out_fp)
