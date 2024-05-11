# standard library imports
from typing import List, Tuple

# 3rd party library imports
from scenedetect import VideoStreamCv2, FrameTimecode
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# local imports
from .ClipDescriptorInterface import ClipDescriptorInterface
from utils.LogHandling.LogHandlers import LogHandlerBase


class ClipDescriptorBase(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorBase, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model = None
        self.__tokenizer = None
        self.__feature_extractor = None  # resizes & normalizes

    def reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def load_models(self):
        self.reload_preferred_device()
        if self.model is None:
            model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model.to(self.preferred_device)
            self.model = model

        if self.__tokenizer is None:
            self.__tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        if self.__feature_extractor is None:
            self.__feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        pixel_values = self.__feature_extractor(torch.stack(frames, dim=0), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.preferred_device)

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values)

        descriptions = self.__tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        descriptions = [description.strip() for description in descriptions]
        return descriptions
