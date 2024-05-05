# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports

# local imports


class CutDetectorInterface(ABC):

    @abstractmethod
    def detect_cuts(self, frames: List) -> List:
        """
        Detects cuts in given serie of frames.

        :return: List of tuples. Every tuple is in form (start_frame_idx, stop_frame_idx) indicating start and end of
            shot/scene.
        """
        pass
