# standard library imports
from io import BytesIO
import os

from typing import List, Tuple

# 3rd party library imports
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, AudioFileClip


# local imports
from StagesProcessor import StagesProcessor, MovieHandlerBase, SceneDetectorBase, ClipDescriptorBase, \
    VoiceSynthesizerBase, MovieComposerBase


DATASETS_FOLDER = "../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset")
CURRENT_DATASET_FOLDER_PROCESSED = CURRENT_DATASET_FOLDER + "_processed"
os.makedirs(CURRENT_DATASET_FOLDER_PROCESSED, exist_ok=True)
FILENAME = "big_buck_bunny_1080p_h264.mov"
FILEPATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)
LANGUAGES = ['pl']


def translate(texts: List[str], target_language: str) -> List[str]:
    translator = GoogleTranslator(source='en', target=target_language)
    return [translator.translate(text=text) for text in texts]


if __name__ == '__main__':
    stages_processor = StagesProcessor(
        movie_handler=MovieHandlerBase(),
        scene_detector=SceneDetectorBase(),
        clip_descriptor=ClipDescriptorBase(),
        voice_synthesizer=VoiceSynthesizerBase(),
        movie_composer=MovieComposerBase()
    )
    scenes = stages_processor.detect_scenes(fp=FILEPATH)
    for language in LANGUAGES:
        print(f"Processing for language: '{language}'")
        descriptions = stages_processor.generate_descriptions(
            fp=FILEPATH,
            scenes=scenes,
            language=language
        )

        # Translation may be needed
        if language != 'en':
            descriptions = translate(texts=descriptions, target_language=language)

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

