# standard library imports
from pathlib import Path
from typing import Union, Tuple, List

import joblib

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from scenedetect import FrameTimecode

# local imports
from . import ClipDescriptorBase
from . import SceneDetectorBase
from . import MovieComposerBase
from . import MovieHandlerBase
from . import VoiceSynthesizerBase
from . import StagesProcessorInterface

CACHE_FOLDER = ".cache/"
mem = joblib.Memory(location=CACHE_FOLDER, verbose=1)


class StagesProcessor(StagesProcessorInterface):

    def __init__(
            self,
            movie_handler: MovieHandlerBase,
            scene_detector: SceneDetectorBase,
            clip_descriptor: ClipDescriptorBase,
            voice_synthesizer: VoiceSynthesizerBase,
            movie_composer: MovieComposerBase
    ):
        super(StagesProcessor, self).__init__()
        self.movie_handler = movie_handler
        self.scene_detector = scene_detector
        self.clip_descriptor = clip_descriptor
        self.voice_synthesizer = voice_synthesizer
        self.movie_composer = movie_composer

        self.load_movie = mem.cache(self.load_movie)
        self.detect_scenes = mem.cache(self.detect_scenes)
        self.generate_descriptions = mem.cache(self.generate_descriptions)
        self.synthesize_descriptions = mem.cache(self.synthesize_descriptions)
        self.compose_movie = mem.cache(self.compose_movie, ignore=['out_fp'])

    def load_movie(self, fp: Union[str, Path]) -> Tuple:
        return self.movie_handler.load(fp=fp)

    def detect_scenes(
            self,
            fp: Union[str, Path],
            *args,
            **kwargs
    ) -> List:
        self.load_movie(fp=fp)
        return self.scene_detector.detect_scenes(video=self.movie_handler.get_video())

    def generate_descriptions(
            self,
            fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]]
    ) -> List:
        self.load_movie(fp=fp)
        self.clip_descriptor.load_models()
        descriptions = self.clip_descriptor.describe(video=self.movie_handler.get_video(), scenes=scenes)
        return descriptions

    def synthesize_descriptions(self, fp: Union[str, Path], descriptions: List[str]) -> List:
        return self.voice_synthesizer.synthesize(texts=descriptions)

    def compose_movie(
            self,
            fp: Union[str, Path],
            video: VideoFileClip,
            audio: AudioFileClip,
            out_fp: Union[str, Path]
    ):
        raise NotImplementedError

