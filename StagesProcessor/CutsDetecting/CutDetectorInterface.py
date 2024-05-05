# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports

# local imports


class CutDetectorInterface(ABC):

    @abstractmethod
    def detect_scenes(self, video) -> List:
        """
        Detects scenes in a given serie of frames.

        :param video: Video that will be processed.
        :return: List of tuples. Every tuple is in form (start_frame_idx, stop_frame_idx) indicating start and end of
            shot/scene.
        """
        pass
