# standard library imports
from io import BytesIO
import os

from typing import List, Tuple

# 3rd party library imports
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, AudioFileClip


# local imports
from StagesProcessor import StagesProcessor
from StagesProcessor.ClipDescribing import (ClipDescriptorViTGPT2, ClipDescriptorLLaVA15, ClipDescriptorGPT4o,
                                            ClipDescriptorLLaVAMistral16, ClipDescriptorLLaVANextVideo34B,
                                            ClipDescriptorVideoLLava)
from StagesProcessor.MovieComposing import MovieComposerBase
from StagesProcessor.MovieHandling import MovieHandlerBase
from StagesProcessor.ScenesDetecting import SceneDetectorBase
from StagesProcessor.VoiceSynthesizing import VoiceSynthesizerBase


DATASETS_FOLDER = "../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset")
CURRENT_DATASET_FOLDER_PROCESSED = CURRENT_DATASET_FOLDER + "_processed"
os.makedirs(CURRENT_DATASET_FOLDER_PROCESSED, exist_ok=True)
FILENAME = "big_buck_bunny_1080p_h264.mov"
FILEPATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)
LANGUAGES = ['en']


def translate(texts: List[str], target_language: str) -> List[str]:
    translator = GoogleTranslator(source='en', target=target_language)
    return [translator.translate(text=text) for text in texts]


if __name__ == '__main__':
    stages_processor = StagesProcessor(
        movie_handler=MovieHandlerBase(),
        scene_detector=SceneDetectorBase(),
        # clip_descriptor=ClipDescriptorLLaVA15(),
        clip_descriptor=ClipDescriptorVideoLLava(),
        # clip_descriptor=ClipDescriptorLLaVAMistral16(),
        # clip_descriptor=ClipDescriptorLLaVANextVideo34B(),
        # clip_descriptor=ClipDescriptorViTGPT2(),
        # clip_descriptor=ClipDescriptorGPT4o(open_ai_key_fp="./keys/open_ai.key"),
        voice_synthesizer=VoiceSynthesizerBase(),
        movie_composer=MovieComposerBase()
    )
    scenes = stages_processor.detect_scenes(fp=FILEPATH)
    english_descriptions = stages_processor.generate_descriptions(
        fp=FILEPATH,
        scenes=scenes,
    )
    for language in LANGUAGES:
        print(f"Processing for language: '{language}'")

        # Translation may be needed
        if language != 'en':
            descriptions = translate(texts=english_descriptions, target_language=language)
        else:
            descriptions = english_descriptions

        synthesized_descriptions = stages_processor.synthesize_descriptions(
            fp=FILEPATH,
            descriptions=descriptions,
            language=language
        )

        stages_processor.compose_movie(
            fp=FILEPATH,
            out_fp=os.path.join(CURRENT_DATASET_FOLDER_PROCESSED, os.path.splitext(FILENAME)[0] + "_" + language + ".mkv"),
            scenes=scenes,
            synthesized_descriptions=synthesized_descriptions
        )

