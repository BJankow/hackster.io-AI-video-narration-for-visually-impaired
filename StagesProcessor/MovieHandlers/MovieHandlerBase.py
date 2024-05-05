# standard library imports
from copy import deepcopy
from pathlib import Path
from typing import Union

# 3rd party library imports
from moviepy.editor import VideoFileClip

# local imports
from .MovieHandlerInterface import MovieHandlerInterface


class MovieHandlerBase(MovieHandlerInterface):

    def __init__(self):
        super(MovieHandlerInterface, self).__init__()
        self.__movie = None

    def load(self, fp: Union[str, Path]):
        if self.__movie is None:
            self.__movie = VideoFileClip(fp)

    def get_movie(self):
        return deepcopy(self.__movie)  # deepcopy because some objects in Python are mutable.

