# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports

# local imports


class ClipDescriptorInterface(ABC):

    @abstractmethod
    def describe(self, frames: List) -> str:
        """
        Generates a text caption for given sequence of frames.

        :param frames: sequence of frames for given clip
        :return: generated caption text.
        """
        pass

