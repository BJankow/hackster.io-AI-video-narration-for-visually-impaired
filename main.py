# standard library imports
from argparse import ArgumentParser
from io import BytesIO
import os
from time import perf_counter_ns
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
from StagesProcessor import TranslatorBase
from StagesProcessor.VoiceSynthesizing import VoiceSynthesizerBase, LANGUAGE2READER
from utils.LogHandling.LogHandlers import StandardLogger

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


if __name__ == '__main__':
    time_start = perf_counter_ns() / 1e9  # [s]

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
        clip_descriptor=ClipDescriptorVideoLLava(),
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
            translator = TranslatorBase(source_language='en', target_language=language)
            narration = translator.batch_translate(texts=english_narration)
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

    logger._logger.info(f"The process took: {perf_counter_ns() / 1e9 - time_start} [s]")

# MEMORY PROBLEMS? MAYBE IT HELPS
# Can't guarantee it will work, but I think you just have to call llama_kv_cache_tokens_rm(ctx, -1, -1); before every new input
