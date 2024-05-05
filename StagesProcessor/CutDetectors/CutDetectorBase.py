# standard library imports
from typing import List

# 3rd party library imports

# local imports
from .CutDetectorInterface import CutDetectorInterface


class CutDetectorBase(CutDetectorInterface):

    def __init__(self):
        super(CutDetectorBase, self).__init__()

    def detect_cuts(self, frames: List) -> List:
        raise NotImplementedError


