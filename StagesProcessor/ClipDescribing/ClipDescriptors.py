# standard library imports
import math
from typing import List, Tuple, Optional
from tqdm import tqdm

# 3rd party library imports
from scenedetect import VideoStreamCv2, FrameTimecode
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, \
    LlavaProcessor, LlavaForConditionalGeneration

# local imports
from .ClipDescriptorInterface import ClipDescriptorInterface
from utils.LogHandling.LogHandlers import LogHandlerBase


class ClipDescriptorViTGPT2(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorViTGPT2, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.__tokenizer: Optional[GPT2TokenizerFast] = None
        self.__processor: Optional[ViTImageProcessor] = None  # resizes & normalizes

    def reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        self.reload_preferred_device()
        if self.model is None:
            model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model.to(self.preferred_device)
            self.model = model

        if self.__tokenizer is None:
            self.__tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        if self.__processor is None:
            self.__processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self.__load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        pixel_values = self.__processor(torch.stack(frames, dim=0), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.preferred_device)

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values)

        descriptions = self.__tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        descriptions = [description.strip() for description in descriptions]
        return descriptions


class ClipDescriptorLLaVA15(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorLLaVA15, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model: Optional[LlavaForConditionalGeneration] = None
        self.__processor: Optional[LlavaProcessor] = None  # resizes & normalizes

    def __reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        # self.__reload_preferred_device()
        if self.model is None:
            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model.to(self.preferred_device)
            self.model = model

        if self.__processor is None:
            self.__processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.__processor.tokenizer.padding_side = "left"

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self.__load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

        descriptions = []
        PER_ITER = 5
        sub_frames_list = [frames[i * PER_ITER: (i+1) * PER_ITER] for i in range(math.ceil(len(frames) / PER_ITER))]
        with torch.no_grad():
            for sub_frames in tqdm(sub_frames_list, desc="Describing images..."):
                inputs = self.__processor(text=[prompt]*len(sub_frames), images=sub_frames, return_tensors="pt")
                output_ids = self.model.generate(**inputs, max_new_tokens=50)
                descriptions += self.__processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

        descriptions = [description.split("ASSISTANT: ")[1] for description in descriptions]
        return descriptions
