# standard library imports
import joblib
from pathlib import Path
from typing import Union, Tuple, List, Optional


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
from . import VoiceSynthesizerInterface
from . import StagesProcessorInterface
from utils.LogHandling.LogHandlers import StandardLogger


CACHE_FOLDER = ".cache/"
mem = joblib.Memory(location=CACHE_FOLDER, verbose=1)


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

        self.load_movie = mem.cache(self.load_movie)
        self.detect_scenes = mem.cache(self.detect_scenes)
        # self.generate_descriptions = mem.cache(self.generate_descriptions, ignore=['scenes'])
        # self.synthesize_descriptions = mem.cache(self.synthesize_descriptions)
        # self.compose_movie = mem.cache(self.compose_movie, ignore=['out_fp', 'scenes', 'synthesized_descriptions'])

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
        """
        Generates descriptions for every shot/scene. Generated descriptions are in english ('en') language.

        :param fp: path to movie file.
        :param scenes: scenes of given movie.
        :return: List of text descriptions.
        """

        self.load_movie(fp=fp)
        descriptions = self.clip_descriptor.describe(video=self.movie_handler.get_video(), scenes=scenes)
        return descriptions

    def convert_descriptions_to_narration(self, descriptions: List[str]) -> List[str]:
        """
        Modifies descriptions so they briefly describe what happened in a video scene by scene. Narrative style.
        :param descriptions: Descriptions - one per scene
        :return: modified descriptions in the form of narrative.
        """
        if self.summarizer is not None:
            return self.summarizer.summarize(sentences=descriptions)
        else:
            self._logger.warning(f"Summarizer is set to {self.summarizer}. "
                                 f"Passing descriptions through without changing.")
            return descriptions

    def synthesize_descriptions(self, fp: Union[str, Path], descriptions: List[str], language: str) -> List:
        return self.voice_synthesizer.synthesize(texts=descriptions, language=language)

    def compose_movie(
            self,
            fp: Union[str, Path],
            out_fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
            synthesized_descriptions: List[AudioSegment]
    ):
        """

        :param fp: path to original movie file.
        :param out_fp: path to file where the composed movie will be saved.
        :param scenes: scenes as Tuple of FrameTimecodes. First indicates beginning of the scene, second - end.
        :param synthesized_descriptions: descriptions as audio.
        :return:
        """
        self.movie_composer.compose(
            video_fp=fp,
            audio_fp=fp,
            scenes=scenes,
            synthesized_descriptions=synthesized_descriptions
        )
        self.movie_composer.save(out_fp=out_fp)

