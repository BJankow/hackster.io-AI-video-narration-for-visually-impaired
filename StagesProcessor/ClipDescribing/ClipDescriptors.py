# standard library imports
import base64
import cv2
import math
from pathlib import Path
import requests
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

# 3rd party library imports
from openai import OpenAI
from PIL import Image
from scenedetect import VideoStreamCv2, FrameTimecode
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, \
    LlavaProcessor, LlavaForConditionalGeneration, AutoProcessor, LlavaNextForConditionalGeneration, \
    LlavaNextProcessor, VideoLlavaProcessor, VideoLlavaForConditionalGeneration

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

        self.__free_memory()
        descriptions = [description.strip() for description in descriptions]
        return descriptions

    def __free_memory(self):
        del self.model
        del self.__processor
        del self.__tokenizer


class ClipDescriptorLLaVA15(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorLLaVA15, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model: Optional[LlavaForConditionalGeneration] = None
        self.__processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self.__desired_data_type = torch.float16
        self.__prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        # self.__prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\nWhat's the content of the image?###Assistant:"
        self.__model_id = "llava-hf/llava-1.5-7b-hf"  # problems
        # self.__model_id = "llava-hf/vip-llava-13b-hf"  # problems
        # self.__model_id = "llava-hf/llava-1.5-13b-hf"  # problems

    def __reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        self.__reload_preferred_device()

        if self.model is None:
            model = LlavaForConditionalGeneration.from_pretrained(
                self.__model_id,
                torch_dtype=self.__desired_data_type,
                # low_cpu_mem_usage=True
            ).to(self.preferred_device)
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            self.model = model

        if self.__processor is None:
            self.__processor = AutoProcessor.from_pretrained(self.__model_id)
            # self.__processor.tokenizer.padding_side = "left"

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self.__load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        descriptions = []
        PER_ITER = 1
        sub_frames_list = [frames[i * PER_ITER: (i+1) * PER_ITER] for i in range(math.ceil(len(frames) / PER_ITER))]
        with torch.no_grad():
            for sub_frames in tqdm(sub_frames_list[:1], desc="Describing images..."):
                inputs = self.__processor(
                    [self.__prompt] * sub_frames.__len__(),
                    sub_frames,
                    # padding=True,
                    return_tensors="pt"
                ).to(self.preferred_device, self.__desired_data_type)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
                descriptions += self.__processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

        self.__free_memory()
        descriptions = [description.split("ASSISTANT: ")[1] for description in descriptions]
        return descriptions

    def __free_memory(self):
        del self.model
        del self.__processor


class ClipDescriptorLLaVAMistral16(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorLLaVAMistral16, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self.__processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self.__desired_data_type = torch.float16
        self.__prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        self.__model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        self.__reload_preferred_device()

        if self.model is None:
            model = (LlavaNextForConditionalGeneration.from_pretrained(
                self.__model_id,
                torch_dtype=self.__desired_data_type,
                # low_cpu_mem_usage=True
            )).to(self.preferred_device)
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            self.model = model

        if self.__processor is None:
            self.__processor = LlavaNextProcessor.from_pretrained(self.__model_id)

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self.__load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        descriptions = []
        PER_ITER = 1
        sub_frames_list = [frames[i * PER_ITER: (i+1) * PER_ITER] for i in range(math.ceil(len(frames) / PER_ITER))]
        with torch.no_grad():
            for sub_frames in tqdm(sub_frames_list, desc="Describing images..."):
                print(f"OBWAZANEK")
                inputs = self.__processor(
                    self.__prompt,
                    sub_frames[0],
                    # padding=True,
                    return_tensors="pt"
                ).to(self.preferred_device)
                print(f"INPUTS: {inputs}")
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100
                )
                descriptions += self.__processor.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                )

        self.__free_memory()
        descriptions = [description.split("ASSISTANT: ")[1] for description in descriptions]
        return descriptions

    def __free_memory(self):
        del self.model
        del self.__processor


class ClipDescriptorLLaVANextVideo34B(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self):
        super(ClipDescriptorLLaVANextVideo34B, self).__init__()
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.model: Optional[LlavaForConditionalGeneration] = None
        self.__processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self.__desired_data_type = torch.float16
        self.__prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        self.__model_id = "llava-hf/LLaVA-NeXT-Video-34B-hf"

    def __reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        self.__reload_preferred_device()

        if self.model is None:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.__model_id,
                torch_dtype=self.__desired_data_type,
                low_cpu_mem_usage=True
            ).to(self.preferred_device)
            self._my_logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            self.model = model

        if self.__processor is None:
            self.__processor = AutoProcessor.from_pretrained(self.__model_id)


class ClipDescriptorGPT4o(ClipDescriptorInterface, LogHandlerBase):
    def __init__(self, open_ai_key_fp: Union[str, Path]):
        super(ClipDescriptorGPT4o, self).__init__()
        self.__open_ai_key_fp = open_ai_key_fp
        self.cpu_device = torch.device("cpu")
        self.preferred_device = torch.device("cpu")
        self.client: Optional[OpenAI] = None

    def __reload_preferred_device(self):
        self.preferred_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __load_models(self):
        # self.__reload_preferred_device()
        with open(self.__open_ai_key_fp, "r") as f:
            key = f.readline().strip('\n')

        if self.client is None:
            self.client = OpenAI(api_key=key)

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self.__load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        base_64_frames_per_scene = []
        for s in tqdm(scenes[1:2], desc="Encoding frames..."):
            # video.seek(s[0])  # beginning of scene will be representation
            scene_frames = []
            while video.frame_number < s[1].frame_num:
                frame = video.read(decode=True, advance=True)
                _, buffer = cv2.imencode(".jpg", frame)
                scene_frames.append(base64.b64encode(buffer).decode("utf-8"))
            base_64_frames_per_scene.append(scene_frames)

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "These are frames from a video scene that I want to depict. Explain what is in the scene shortly.",
                    *map(lambda x: {"image": x, "resize": 768}, base_64_frames)
                ]
            } for base_64_frames in base_64_frames_per_scene
        ]
        # You are assistant that describes

        descriptions = self.client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-3.5-turbo",
            messages=prompt_messages,
            max_tokens=300
        )

        self.__free_memory()
        return descriptions

    def __free_memory(self):
        del self.client



