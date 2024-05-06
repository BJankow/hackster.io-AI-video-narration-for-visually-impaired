# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports

# local imports


class ClipDescriptorInterface(ABC):

    @abstractmethod
    def describe(self, video, scenes: List) -> List[str]:
        """
        Generates a text caption for given sequence of frames.

        :param video: Video with frames described.
        :param scenes: List of sequence of frames for given clip. List of clips as an input.
        :return: List of generated caption texts. One text for every clip.
        """
        pass

