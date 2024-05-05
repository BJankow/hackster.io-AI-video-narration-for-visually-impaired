# standard library imports
from typing import Any

# 3rd party library imports

# local imports
from .VoiceSynthesizerInterface import VoiceSynthesizerInterface


class VoiceSynthesizerBase(VoiceSynthesizerInterface):

    def __init__(self):
        super(VoiceSynthesizerBase, self).__init__()

    def synthesize(self, text: str) -> Any:
        raise NotImplementedError

