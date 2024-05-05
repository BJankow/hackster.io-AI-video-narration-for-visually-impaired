# standard library imports
from pathlib import Path
from typing import Union

# 3rd party library imports

# local imports
from . import ClipDescriptorBase
from . import CutDetectorBase
from . import MovieComposerBase
from . import MovieHandlerBase
from . import VoiceSynthesizerBase
from . import StagesProcessorInterface


class StagesProcessor(StagesProcessorInterface):

    def __init__(
            self,
            movie_handler: MovieHandlerBase,
            cut_detector: CutDetectorBase,
            clip_descriptor: ClipDescriptorBase,
            voice_synthesizer: VoiceSynthesizerBase,
            movie_composer: MovieComposerBase
    ):
        super(StagesProcessor, self).__init__()
        self.movie_handler = movie_handler
        self.cut_detector = cut_detector
        self.clip_descriptor = clip_descriptor
        self.voice_synthesizer = voice_synthesizer
        self.movie_composer = movie_composer

    def load_movie(self, fp: Union[str, Path]):
        self.movie_handler.load(fp=fp)

    def detect_cuts(self, *args, **kwargs):
        self.cut_detector.detect_cuts(frames=self.movie_handler.get_movie())

    def generate_descriptions(self):
        raise NotImplementedError

    def synthesize_descriptions(self):
        raise NotImplementedError

    def compose_movie(self, out_fp: Union[str, Path]):
        raise NotImplementedError

