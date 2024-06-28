# standard library imports
from typing import List

# 3rd party library imports

# local imports
from .SummarizationInterface import SummarizerInterface
from utils.LogHandling.LogHandlers import StandardLogger


class SummarizerBase(SummarizerInterface, StandardLogger):

    def __init__(self):
        super(SummarizerBase).__init__()

    def summarize(
            self,
            sentences: List[str]
    ) -> List[str]:
        """
        This function expects input sentences to be description of movie shots. It will modify these descriptions to
            provide a short summary in a narrative style. The summary should be concise and clear.
        :param sentences: List of sentences that you want to summarize/modify.
        :return: List of modified sentences forming a nice concise and clear summary.
        """

        raise NotImplementedError
