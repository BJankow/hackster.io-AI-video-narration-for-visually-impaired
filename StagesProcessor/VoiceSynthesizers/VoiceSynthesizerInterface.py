# standard library imports
from abc import ABC, abstractmethod
from typing import Any

# 3rd party library imports

# local imports


class VoiceSynthesizerInterface(ABC):

    @abstractmethod
    def synthesize(self, text: str) -> Any:  # TODO: change Any into something more specific (sound representation type)
        """
        Synthesizes given text into voice

        :param text: text that will be synthesized.
        :return: None.
        """
        pass

