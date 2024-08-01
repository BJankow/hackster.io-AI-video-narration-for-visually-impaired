# standard library imports
from typing import List, Iterable

# 3rd party library imports
from deep_translator import GoogleTranslator

# local imports
from .TranslatorInterface import TranslatorInterface


class TranslatorBase(TranslatorInterface):
    def __init__(
            self,
            source_language: str,
            target_language: str
    ):
        super(TranslatorBase, self).__init__()
        self.source_language = source_language
        self.target_language = target_language
        self.translator = GoogleTranslator(source=source_language, target=target_language)

    def translate(self, text: str) -> str:
        return self.translator.translate(text=text)

    def batch_translate(self, texts: Iterable[str]) -> List[str]:
        return [self.translate(text=text) for text in texts]
