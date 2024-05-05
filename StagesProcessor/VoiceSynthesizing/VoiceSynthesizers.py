# standard library imports
from typing import Any

# 3rd party library imports
from pydub.silence import detect_silence
import pyttsx3

# local imports
from .VoiceSynthesizerInterface import VoiceSynthesizerInterface


class VoiceSynthesizerBase(VoiceSynthesizerInterface):

    def __init__(self):
        super(VoiceSynthesizerBase, self).__init__()
        # self.__engine = pyttsx3.init()

    def synthesize(self, text: str) -> Any:
        pass

