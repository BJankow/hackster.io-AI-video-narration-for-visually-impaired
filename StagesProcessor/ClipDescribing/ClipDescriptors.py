# standard library imports
import base64
import cv2
import math
from pathlib import Path

import numpy as np
import psutil
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
from utils.LogHandling.LogHandlers import StandardLogger


class ClipDescriptorBase(ClipDescriptorInterface, StandardLogger):
    def __init__(self):
        super(ClipDescriptorBase, self).__init__()
        self.preferred_device = torch.device("cpu")
        self._desired_data_type = torch.float16
        self._model_id = None

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

    def describe(
            self,
            video: VideoStreamCv2,
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
    ) -> List[str]:
        raise NotImplementedError


class ClipDescriptorViTGPT2(ClipDescriptorBase):
    def __init__(self):
        super(ClipDescriptorViTGPT2, self).__init__()
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._tokenizer: Optional[GPT2TokenizerFast] = None
        self._processor: Optional[ViTImageProcessor] = None  # resizes & normalizes
        self._model_id = "nlpconnect/vit-gpt2-image-captioning"

    def _load_models(self):
        """
        Instantiates LLM model in memory.

        :return:
        """
        self._reload_preferred_device()
        if self.model is None:
            model = VisionEncoderDecoderModel.from_pretrained(
                self._model_id,
                torch_dtype=self._desired_data_type,
                device_map=self.preferred_device,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            )
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self._model_id)

        if self._processor is None:
            self._processor = ViTImageProcessor.from_pretrained(self._model_id)

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self._load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        pixel_values = self._processor(torch.stack(frames, dim=0), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.preferred_device)

        with torch.inference_mode():
            output_ids = self.model.generate(pixel_values)

        descriptions = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        self._free_memory()
        descriptions = [description.strip() for description in descriptions]
        return descriptions

    def _free_memory(self):
        del self.model
        del self._processor
        del self._tokenizer


class ClipDescriptorLLaVA15(ClipDescriptorBase):
    def __init__(self):
        super(ClipDescriptorLLaVA15, self).__init__()
        self.model: Optional[LlavaForConditionalGeneration] = None
        self._processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self._prompt = "USER: <image>\nWhat's the content of the image?\nASSISTANT:"
        # style_examples = (f"STYLE EXAMPLES:\n"
        #                   "- Sunrise over the African savanna.\n"
        #                   "- Animals gather at Pride Rock.\n"
        #                   "- Mufasa and Sarabi present their newborn son, Simba.\n"
        #                   "- Rafiki blesses him and lifts him up.\n"
        #                   "- They cheer and bow.\n"
        #                   "- The Pride Lands under the rising sun.\n"
        #                   # "- A green field with blue sky.\n"
        #                   # "- A house with several chairs and a woman standing in the middle of the living room. She is smiling\n"
        #                   # "- Ocean with a small ship sailing though it.\n"
        #                   "\n")
        # self._prompt = (f"{style_examples}"
        #                 "CURRENT SCENE: <image>\n"
        #                 "TASK: Directly describe the scene without starting with auxiliary words or helping verbs. "
        #                 "Use pronouns (like 'it', 'she', 'he') where appropriate to avoid repetition. "
        #                 "Your description should be brief, and collect only the most important facts.\n"
        #                 "DESCRIPTION:")
        self._model_id = "llava-hf/llava-1.5-7b-hf"

    def _load_models(self):
        """
        Instantiates LLM model in memory.

        :return:
        """
        self._reload_preferred_device()

        if self.model is None:
            model = LlavaForConditionalGeneration.from_pretrained(
                self._model_id,
                device_map=self.preferred_device,
                torch_dtype=self._desired_data_type,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            )
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            # self._processor.tokenizer.padding_side = "left"

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self._load_models()
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        descriptions = []
        PER_ITER = 8
        sub_frames_list = [frames[i * PER_ITER: (i+1) * PER_ITER] for i in range(math.ceil(len(frames) / PER_ITER))]
        with torch.inference_mode():
            for sub_frames in tqdm(sub_frames_list, desc="Describing images..."):
                inputs = self._processor(
                    [self._prompt] * sub_frames._len_(),
                    sub_frames,
                    # padding=True,
                    return_tensors="pt"
                ).to(self.preferred_device, self._desired_data_type)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
                descriptions += self._processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    # clean_up_tokenization_spaces=False
                )

        self._free_memory()
        descriptions = [description.split("ASSISTANT: ")[1] for description in descriptions]
        return descriptions

    def describe_single_image(self, image: np.array) -> str:
        """
        Given image creates description using loaded LLM model.

        :return: description in form of string
        """
        self._load_models()

        with torch.inference_mode():
            inputs = self._processor(
                self._prompt,
                image,
                # padding=True,
                return_tensors="pt"
            ).to(self.preferred_device, self._desired_data_type)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
            description = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=False
            )[0]

        self._free_memory()
        return description

    def _free_memory(self):
        del self.model
        del self._processor


class ClipDescriptorVideoLLava(ClipDescriptorBase):
    def __init__(self):
        super(ClipDescriptorVideoLLava, self).__init__()
        self.model: Optional[VideoLlavaForConditionalGeneration] = None
        self._processor: Optional[VideoLlavaProcessor] = None  # resizes & normalizes
        self._scene_descriptions = {}
        self._model_id = "LanguageBind/Video-LLaVA-7B-hf"

    def _prompt(self) -> str:
        """
        Generates prompt.

        :return: prompt in form of a string.
        """
        # https://www.reddit.com/r/LocalLLaMA/comments/1asyo9m/llava_16_how_to_write_proper_prompt_that_will/
        # (FinancialNailer answer is helpful to get rid of repeating "The image/video shows...")
        #
        scenes_string = ""
        style_examples = (f"STYLE EXAMPLES:\n"
                          "- Sunrise over the African savanna.\n"
                          "- Animals gather at Pride Rock.\n"
                          "- Mufasa and Sarabi present their newborn son, Simba.\n"
                          "- Rafiki blesses him and lifts him up.\n"
                          "- They cheer and bow.\n"
                          "- The Pride Lands under the rising sun.\n"
                          "\n")

        for idx, description in list(self._scene_descriptions.items())[-5:]:  # consider only N last descriptions
            scenes_string += f"\nScene {idx + 1}: {description}"

        # prompt = ("USER: <video>\n"
        #           "Describe the video scene briefly (in a laconic way), only the most important facts.\n"
        #           "Skip auxiliary words and helping verbs.\n"
        #           "NARRATION:")
        if scenes_string == "":
            # prompt = "USER: <video>\nWhat's the content of the video?\nDESCRIPTION:"
            prompt = (f""
                      # f"{style_examples}"
                      "CURRENT SCENE: <video>\n"
                      "TASK: Directly describe the scene without starting with auxiliary words or helping verbs. "
                      # "Use pronouns (like 'it', 'she', 'he') where appropriate to avoid repetition. "
                      # "Prefer using complex sentences instead of several simple sentences. "
                      # "The style of your description should be similar to STYLE EXAMPLES given above."
                      # "Ommit describing colors. "
                      # "Description should be brief, and collect only the most important facts.\n"
                      "DESCRIPTION:")
        else:
            prompt = (f"{style_examples}"
                      f"PREVIOUS SCENE DESCRIPTIONS:{scenes_string}\n\n"
                      "CURRENT SCENE:\n<video>\n"
                      "TASK: Directly describe the scene without starting with auxiliary words or helping verbs. "
                      "Use pronouns (like 'it', 'she', 'he') where appropriate to avoid repetition. "
                      "The style of your description should be similar to STYLE EXAMPLES given above."
                      "Your description should be brief, and collect only the most important facts."
                      "Keep in mind PREVIOUS SCENE DESCRIPTIONS given above - your description should try to be coherent continuation of the narration\n"
                      "DESCRIPTION:")

        return prompt

    def _load_models(self):
        """
        Instantiates LLM model in memory.

        :return:
        """
        self._reload_preferred_device()

        if self.model is None:
            model = (VideoLlavaForConditionalGeneration.from_pretrained(
                self._model_id,
                device_map=self.preferred_device,
                torch_dtype=self._desired_data_type,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            ))
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

        if self._processor is None:
            self._processor = VideoLlavaProcessor.from_pretrained(self._model_id)

    def describe(
            self,
            video: VideoStreamCv2,
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
    ) -> List[str]:
        self._load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning

        s_idx = 0
        with torch.inference_mode():
            for s in tqdm(scenes, desc="Describing clips..."):
                clip = []
                chosen_frames = np.linspace(start=s[0].frame_num, stop=s[1].frame_num, num=10, dtype=int)
                for c_f in chosen_frames[1:-1]:
                    video.seek(int(c_f))
                    frame = video.read()[:, :, ::-1].copy()
                    # cv2.imshow('a', frame)
                    # cv2.waitKey()
                    # cv2.destroyWindow('a')
                    clip.append(torch.from_numpy(frame))  # BGR2RGB conversion
                clip = torch.stack(clip)
                inputs = self._processor(
                    text=self._prompt(),
                    videos=clip,
                    # padding=True,
                    return_tensors="pt"
                ).to(self.preferred_device, self._desired_data_type)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                current_descriptions = self._processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].split("DESCRIPTION: ")[1]
                if current_descriptions.__len__() > 0:
                    if not current_descriptions[-1].endswith('.'):  # not finished
                        current_descriptions = '.'.join(current_descriptions.split('.')[:-1]) + '.'
                self._scene_descriptions.update({
                    s_idx: current_descriptions
                })
                s_idx += 1

        self._free_memory()
        return list(self._scene_descriptions.values())

    def _free_memory(self):
        del self.model
        del self._processor


class ClipDescriptorLLaVAMistral16(ClipDescriptorBase):
    def __init__(self):
        super(ClipDescriptorLLaVAMistral16, self).__init__()
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self._processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self._prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        self._model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    def _load_models(self):
        """
        Instantiates LLM model in memory.

        :return:
        """
        self._reload_preferred_device()

        if self.model is None:
            model = (LlavaNextForConditionalGeneration.from_pretrained(
                self._model_id,
                device_map=self.preferred_device,
                torch_dtype=self._desired_data_type,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            )).to(self.preferred_device)
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

        if self._processor is None:
            self._processor = LlavaNextProcessor.from_pretrained(self._model_id)

    def describe(self, video: VideoStreamCv2, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> List[str]:
        self._load_models()
        # pick frame - take frame that is in 10% from beginning
        video.reset()  # make sure video is at the beginning
        frames = []
        for s in scenes:
            video.seek(s[0])  # beginning of scene will be representation
            frames.append(torch.from_numpy(video.read(decode=True, advance=True)))

        descriptions = []
        PER_ITER = 1
        sub_frames_list = [frames[i * PER_ITER: (i+1) * PER_ITER] for i in range(math.ceil(len(frames) / PER_ITER))]
        with torch.inference_mode():
            for sub_frames in tqdm(sub_frames_list, desc="Describing images..."):
                inputs = self._processor(
                    self._prompt,
                    sub_frames[0],
                    # padding=True,
                    return_tensors="pt"
                ).to(self.preferred_device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100
                )
                descriptions += self._processor.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                )

        self._free_memory()
        descriptions = [description.split("ASSISTANT: ")[1] for description in descriptions]
        return descriptions

    def _free_memory(self):
        del self.model
        del self._processor


class ClipDescriptorLLaVANextVideo34B(ClipDescriptorBase):
    def __init__(self):
        super(ClipDescriptorLLaVANextVideo34B, self).__init__()
        self.model: Optional[LlavaForConditionalGeneration] = None
        self._processor: Optional[LlavaProcessor] = None  # resizes & normalizes
        self._prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        self._model_id = "llava-hf/LLaVA-NeXT-Video-34B-hf"

    def _load_models(self):
        """
        Instantiates LLM model in memory.

        :return:
        """
        self._reload_preferred_device()

        if self.model is None:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self._model_id,
                device_map=self.preferred_device,
                torch_dtype=self._desired_data_type,
                low_cpu_mem_usage=True  # requires Accelerate version >= 0.9.0
            ).to(self.preferred_device)
            self._logger.debug(model)  # TODO - it is not visible in terminal right now...
            model.eval()
            model = torch.compile(model, mode='reduce-overhead')  # speeds up inference. torch >= 2.0.
            self.model = model

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self._model_id)





