# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports
from pydub.audio_segment import AudioSegment

# local imports


class VoiceSynthesizerInterface(ABC):

    @abstractmethod
    def synthesize(self, texts: List[str], language: str) -> List[AudioSegment]:
        """
        Synthesizes given text into voice

        :param texts: List of texts that will be synthesized.
        :param language: Selected language.
        :return: None.
        """
        pass

