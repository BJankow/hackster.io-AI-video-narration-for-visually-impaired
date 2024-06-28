# standard library imports
from abc import ABC, abstractmethod
from typing import List

# 3rd party library imports

# local imports


class SummarizerInterface:

    @abstractmethod
    def summarize(
            self,
            sentences: List[str]
    ) -> List[str]:
        """
        This function expects input sentences to be description of movie shots. It will modify these descriptions to
            provide a short summary in a narrative style. The summary should be concise and clear.
        :param sentences: List of sentences that you want to summarize/modify.
        :return:
        """