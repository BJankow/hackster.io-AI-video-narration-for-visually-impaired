# standard library imports
from pathlib import Path
from typing import Union, Tuple, List, Optional, Iterable

import numpy as np
# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub.audio_segment import AudioSegment
from scenedetect import FrameTimecode


# local imports
from . import ClipDescriptorInterface
from . import SceneDetectorInterface
from . import SummarizerInterface
from . import MovieComposerInterface
from . import MovieHandlerInterface
from . import TranslatorInterface
from . import VoiceSynthesizerInterface
from . import StagesProcessorInterface
from utils.LogHandling.LogHandlers import StandardLogger


class StagesProcessor(StagesProcessorInterface, StandardLogger):

    def __init__(
            self,
            movie_handler: MovieHandlerInterface,
            scene_detector: SceneDetectorInterface,
            clip_descriptor: ClipDescriptorInterface,
            voice_synthesizer: VoiceSynthesizerInterface,
            movie_composer: MovieComposerInterface,
            summarizer: Optional[SummarizerInterface] = None,
    ):
        super(StagesProcessor, self).__init__()
        self.movie_handler = movie_handler
        self.scene_detector = scene_detector
        self.clip_descriptor = clip_descriptor
        self.voice_synthesizer = voice_synthesizer
        self.movie_composer = movie_composer
        self.summarizer = summarizer

    def load_movie(self, fp: Union[str, Path]) -> Tuple:
        self._logger.info("Loading movie...")
        return self.movie_handler.load(fp=fp)

    def detect_scenes(
            self,
            fp: Union[str, Path],
            time_stop=np.inf,
            time_start=0.0,
            *args,
            **kwargs
    ) -> List:
        self._logger.info("Detecting scenes...")
        self.load_movie(fp=fp)
        return self.scene_detector.detect_scenes(
            video=self.movie_handler.get_video(),
            time_stop=time_stop,
            time_start=time_start,
        )

    def generate_descriptions(
            self,
            fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
    ) -> List:
        self._logger.info("Generating descriptions...")
        self.load_movie(fp=fp)
        descriptions = self.clip_descriptor.describe(
            video=self.movie_handler.get_video(),
            scenes=scenes,
        )
        return descriptions

    def convert_descriptions_to_narration(self, descriptions: List[str]) -> List[str]:
        self._logger.info("Converting descriptions to narration...")
        if self.summarizer is not None:
            return self.summarizer.summarize(sentences=descriptions)
        else:
            self._logger.warning(f"Summarizer is set to {self.summarizer}. "
                                 f"Passing descriptions through without changing.")
            return descriptions

    def synthesize_descriptions(
            self,
            fp: Union[str, Path],
            descriptions: List[str],
            language: str
    ) -> List:
        self._logger.info("Synthesizing descriptions...")
        return self.voice_synthesizer.synthesize(texts=descriptions, language=language)

    def compose_movie(
            self,
            fp: Union[str, Path],
            out_fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
            synthesized_descriptions: List[AudioSegment]
    ):
        self._logger.info("Composing movie...")
        self.movie_composer.compose(
            video_fp=fp,
            audio_fp=fp,
            scenes=scenes,
            synthesized_descriptions=synthesized_descriptions
        )
        self.movie_composer.save(out_fp=out_fp)

