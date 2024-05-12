# standard library imports
from io import BytesIO
import os.path
from pathlib import Path

# 3rd party library imports
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub import AudioSegment

# local imports

FILE_DIR_MOVIES = os.path.join(Path.home(), 'Projects', 'datasets', 'hackster.io-AI-video-narration-for-visually-impaired-dataset')
MOVIE_FILE = os.path.join(FILE_DIR_MOVIES, "big_buck_bunny_1080p_h264.mov")

FILE_DIR_LECTORS = os.path.join(Path.home(), 'Projects', 'datasets', 'hackster.io-AI-video-narration-for-visually-impaired-lectors')
FILE_LECTOR = os.path.join(FILE_DIR_LECTORS, "test.wav")

if __name__ == '__main__':
    audio = AudioSegment.from_file(MOVIE_FILE)
    buffer = BytesIO()
    audio.export(buffer, format="wav")

    pipeline = VoiceActivityDetection(segmentation="anilbs/segmentation")
    HYPER_PARAMETERS = {
        # onset/offset activation thresholds
        "onset": 0.95, "offset": 0.3,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(buffer)
    a = [(el.start, el.end) for el in vad._tracks._list._lists[0]]
    pass

