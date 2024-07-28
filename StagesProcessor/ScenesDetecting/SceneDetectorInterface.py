# standard library imports
from abc import ABC, abstractmethod
from typing import List, Tuple

# 3rd party library imports
from scenedetect import FrameTimecode

# local imports


class SceneDetectorInterface(ABC):

    @abstractmethod
    def detect_scenes(
            self,
            video,
            time_stop: float,
            time_start: float
    ) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        Detects scenes in a given serie of frames.

        :param video: Video that will be processed.
        :param time_stop: (seconds) in what moment of the movie to stop processing.
        :param time_start: (seconds) in what moment of the movie to start processing.
        :return: List of tuples. Every tuple is in form (start_FrameTimecode, stop_FrameTimecode) indicating start and
            end of every shot/scene.
        """
