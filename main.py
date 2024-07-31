# standard library imports
from argparse import ArgumentParser
from io import BytesIO
import os

from typing import List, Tuple, Optional, Set

# 3rd party library imports
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, AudioFileClip


# local imports
from StagesProcessor import StagesProcessor
from StagesProcessor.ClipDescribing import (ClipDescriptorViTGPT2, ClipDescriptorLLaVA15,
                                            ClipDescriptorLLaVAMistral16, ClipDescriptorLLaVANextVideo34B,
                                            ClipDescriptorVideoLLava)
from StagesProcessor.MovieComposing import MovieComposerBase
from StagesProcessor.MovieHandling import MovieHandlerBase
from StagesProcessor.ScenesDetecting import SceneDetectorBase
from StagesProcessor.VoiceSynthesizing import VoiceSynthesizerBase, LANGUAGE2READER
from utils.LogHandling.LogHandlers import StandardLogger

# FILENAME = "big_buck_bunny_1080p_h264.mov"
# FILENAME = "Episode 1 - Winter is Coming.mp4"
# FILENAME = "02 - Samotny cyborg.mp4"
# FILENAME = "Spirit.Stallion.of.the.Cimarron.2002.1080p.BluRay.DDP.5.1.x265-EDGE2020.mkv"

t = GoogleTranslator()
supported_languages = t.get_supported_languages(as_dict=True)
AVAILABLE_TRANSLATOR_LANGUAGES = list(supported_languages.keys())
AVAILABLE_TRANSLATOR_ABBREVIATIONS = list(supported_languages.values())
AVAILABLE_LANGUAGES_IN_GOOGLE_TRANSLATOR = set(AVAILABLE_TRANSLATOR_LANGUAGES + AVAILABLE_TRANSLATOR_ABBREVIATIONS)

# Verify if LANGUAGE2READER.keys() are setup properly
for language in LANGUAGE2READER.keys():
    if language not in AVAILABLE_LANGUAGES_IN_GOOGLE_TRANSLATOR:
        raise ValueError(
            f"Language: {language} configured as a key in LANGUAGE2READER dict from "
            f"StagesProcessor/VoiceSynthesizing/VoiceSynthesizers.py file is not available in "
            f"deep_translator.GoogleTranslator. "
            f"Available languages in deep_translator.GoogleTranslator: {AVAILABLE_LANGUAGES_IN_GOOGLE_TRANSLATOR}"
        )

# Intersection of languages available for translator and prepared voice samples.
AVAILABLE_LANGUAGES = list(set(LANGUAGE2READER.keys()) & AVAILABLE_LANGUAGES_IN_GOOGLE_TRANSLATOR)


def translate(texts: List[str], target_language: str) -> List[str]:
    translator = GoogleTranslator(source='en', target=target_language)
    return [translator.translate(text=text) for text in texts]


if __name__ == '__main__':
    logger = StandardLogger()
    parser = ArgumentParser()
    parser.add_argument(
        "--fp",
        help="path to movie file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-dir",
        help="Directory for processed movie. Default: same directory as the input file.",
        type=Optional[str],
        default=None,
        required=False
    )
    parser.add_argument(
        "--languages",
        help="Choose narrator's language. If more than one given Multiple output files will be create: 1 per language. "
             "Available: ['en', 'pl']",
        action="append",
        default=[],
        required=False
    )

    args = parser.parse_args()
    fp = args.fp
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File: {fp} does not exist.")

    out_dir: Optional[str] = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(fp)  # same folder as input file
    os.makedirs(out_dir, exist_ok=True)  # creating output directory
    logger._logger.info(f"Output directory: {os.path.abspath(out_dir)}")

    languages: Set[str] = set(args.languages)
    if languages.__len__() == 0:
        languages = {'en'}
    print(f"{languages=}")
    for language in languages:
        if language not in AVAILABLE_LANGUAGES:
            raise ValueError(f"{language=} is not valid, {AVAILABLE_LANGUAGES=}.")

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
    scenes = stages_processor.detect_scenes(
        fp=fp,
        # time_start=0.0,  # [s]
        # time_stop=90.0  # [s]
    )
    english_descriptions = stages_processor.generate_descriptions(
        fp=fp,
        scenes=scenes,
    )

    english_narration = stages_processor.convert_descriptions_to_narration(
        descriptions=english_descriptions
    )

    for language in languages:
        print(f"Processing for language: '{language}'")

        # Translation may be needed
        if language != 'en':
            narration = translate(texts=english_narration, target_language=language)
        else:
            narration = english_narration

        synthesized_descriptions = stages_processor.synthesize_descriptions(
            fp=fp,
            descriptions=narration,
            language=language
        )

        stages_processor.compose_movie(
            fp=fp,
            out_fp=os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(fp))[0] + "_" + language + os.path.splitext(os.path.basename(fp))[1]
            ),
            scenes=scenes,
            synthesized_descriptions=synthesized_descriptions
        )

# MEMORY PROBLEMS? MAYBE IT HELPS
# Can't guarantee it will work, but I think you just have to call llama_kv_cache_tokens_rm(ctx, -1, -1); before every new input
