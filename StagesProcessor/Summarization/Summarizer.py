# standard library imports
from typing import List, Optional

# 3rd party library imports
import torch
from transformers import PegasusModel

# local imports
from .SummarizationInterface import SummarizerInterface
from utils.LogHandling.LogHandlers import StandardLogger


class SummarizerBase(SummarizerInterface, StandardLogger):

    def __init__(self):
        super(SummarizerBase).__init__()
        self.preferred_device = torch.device("cpu")
        self._desired_data_type = torch.float16

    def summarize(
            self,
            sentences: List[str]
    ) -> List[str]:
        raise NotImplementedError

    def _reload_preferred_device(self):
        """
        Basing on CUDA device availability sets device on which calculation will be done (CPU or GPU).

        :return:
        """
        if torch.cuda.is_available():
            self.preferred_device = torch.device("cuda:0")
            self._logger.info(f"Utilized (GPU): {torch.cuda.get_device_name(self.preferred_device)}")
        else:
            self.preferred_device = torch.device("cpu")
            self._logger.info(f"Utilized device - CPU")


class PegasusSummarizer(SummarizerBase):
    def __init__(self):
        super(PegasusSummarizer, self).__init__()
        self.model = None
        self.__model_id = "google/pegasus-x-large"

    def _load_models(self):
        """
        Loads models to memory of chosen device.
        :return:
        """

        # https://www.reddit.com/r/LocalLLaMA/comments/169jr7f/best_model_for_summarization_task/
        # https://huggingface.co/docs/transformers/perf_torch_compile
        # https://huggingface.co/docs/transformers/big_models
        # https://huggingface.co/docs/transformers/perf_infer_cpu
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one
        # https://huggingface.co/docs/transformers/serialization
        # https://huggingface.co/docs/transformers/multilingual
        # https://huggingface.co/docs/transformers/tasks/prompting
        # https://huggingface.co/docs/transformers/model_doc/pegasus_x

        self._reload_preferred_device()

        if self.model is None:
            model = (PegasusModel.from_pretrained(
                self.__model_id,
                device_map=self.preferred_device,
                torch_dtype=self._desired_data_type,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            ))
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

    def summarize(
            self,
            sentences: List[str]
    ) -> List[str]:
        raise NotImplementedError
