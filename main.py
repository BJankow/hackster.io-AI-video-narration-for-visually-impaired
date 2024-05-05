# standard library imports
import os

# 3rd party library imports

# local imports
from StagesProcessor import StagesProcessor, MovieHandlerBase, CutDetectorBase, ClipDescriptorBase, \
    VoiceSynthesizerBase, MovieComposerBase


DATASETS_FOLDER = "../../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset/")
FILENAME = "big_buck_bunny_1080p_h264.mov"
FILEPATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)


if __name__ == '__main__':
    stages_processor = StagesProcessor(
        movie_handler=MovieHandlerBase(),
        cut_detector=CutDetectorBase(),
        clip_descriptor=ClipDescriptorBase(),
        voice_synthesizer=VoiceSynthesizerBase(),
        movie_composer=MovieComposerBase()
    )
    stages_processor.load_movie(fp=FILEPATH)
    stages_processor.detect_cuts()

