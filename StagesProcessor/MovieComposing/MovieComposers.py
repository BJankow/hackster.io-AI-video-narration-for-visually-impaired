# standard library imports
from pathlib import Path
from typing import List, Union

# 3rd party library imports

# local imports
from .MovieComposerInterface import MovieComposerInterface


class MovieComposerBase(MovieComposerInterface):

    def __init__(self):
        super(MovieComposerBase, self).__init__()

    def compose(self, frames: List, sound):
        raise NotImplementedError

    def save(self, out_fp: Union[str, Path]):
        raise NotImplementedError
