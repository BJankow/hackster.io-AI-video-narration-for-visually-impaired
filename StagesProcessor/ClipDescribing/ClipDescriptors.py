# standard library imports
from typing import List

# 3rd party library imports

# local imports
from .ClipDescriptorInterface import ClipDescriptorInterface


class ClipDescriptorBase(ClipDescriptorInterface):
    def __init__(self):
        super(ClipDescriptorBase, self).__init__()

    def describe(self, frames: List) -> str:
        print("AAA")
        raise NotImplementedError

