# standard library imports
from abc import ABC, abstractmethod
from typing import Iterable, List

# 3rd party library imports

# local imports


class TranslatorInterface(ABC):

    @abstractmethod
    def translate(self, text: str) -> str:
        """
        Performs translation task.

        :param text: text in particular language to be translated to other language.
        :return: translated text.
        """

    @abstractmethod
    def batch_translate(self, texts: Iterable[str]) -> List[str]:
        """
        Performs multiple translation task. Source and target language were defined during instance initialization.

        :param texts: Iterable of texts to be translated from particular language to target language.
        :return: list of translated texts.
        """

