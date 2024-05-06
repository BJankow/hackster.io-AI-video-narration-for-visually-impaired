# standard library imports
import os

# 3rd party library imports

# local imports
from StagesProcessor import StagesProcessor, MovieHandlerBase, SceneDetectorBase, ClipDescriptorBase, \
    VoiceSynthesizerBase, MovieComposerBase


DATASETS_FOLDER = "../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset/")
FILENAME = "big_buck_bunny_1080p_h264.mov"
FILEPATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)


if __name__ == '__main__':
    stages_processor = StagesProcessor(
        movie_handler=MovieHandlerBase(),
        scene_detector=SceneDetectorBase(),
        clip_descriptor=ClipDescriptorBase(),
        voice_synthesizer=VoiceSynthesizerBase(),
        movie_composer=MovieComposerBase()
    )
    video, audio = stages_processor.load_movie(fp=FILEPATH)
    scenes = stages_processor.detect_scenes(fp=FILEPATH)
    descriptions = stages_processor.generate_descriptions(fp=FILEPATH, scenes=scenes)
    synthesized_descriptions = stages_processor.synthesize_descriptions(fp=FILEPATH, descriptions=descriptions)

