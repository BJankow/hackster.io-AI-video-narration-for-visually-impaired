# standard library imports
from concurrent.futures import ThreadPoolExecutor, wait
from io import BytesIO
import os
import os.path
from pathlib import Path
from typing import List
from tqdm import tqdm

os.environ["COQUI_TOS_AGREED"] = "1"

# 3rd party library imports
from gtts import gTTS
import numpy as np
from pydub.audio_segment import AudioSegment
from pydub.playback import play
from pydub.silence import detect_silence
import torch
from TTS.api import TTS
from TTS.utils.audio import AudioProcessor

import pyttsx3

# local imports
from .VoiceSynthesizerInterface import VoiceSynthesizerInterface

READERS_FOLDER = os.path.join(Path.home(), 'Projects', 'datasets', 'hackster.io-AI-video-narration-for-visually-impaired-lectors')
LANGUAGE2READER = {
    'en': os.path.join(READERS_FOLDER, 'AlanWatts.mp3'),
    # 'pl': os.path.join(READERS_FOLDER, 'Knapik.mp3'),
    'pl': os.path.join(READERS_FOLDER, 'DariuszSzpakowski.mp3')
    # 'pl': os.path.join(READERS_FOLDER, 'MagdalenaSchejbal.mp3')
}


class VoiceSynthesizerBase(VoiceSynthesizerInterface):

    def __init__(self):
        super(VoiceSynthesizerBase, self).__init__()
        # self.__engine = pyttsx3.init()

    def synthesize(self, texts: List[str], language: str) -> List[AudioSegment]:

        # synthesized_texts = [gTTS(text=text, lang='en', slow=False) for text in texts]

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True, progress_bar=True)
        tts.eval()
        synthesized_texts = [tts.tts(
            text=text,
            # speaker_wav=LANGUAGE2READER[language],
            language=language,
            speed=1.3,
            emotion="happy"
        ) for text in texts]
        audio_segments = []

        def inner(idx, s_t):

            # mp3_fp = BytesIO()  # write to buffer
            # audio_processor.save_wav(wav=s_t, pipe_out=mp3_fp)
            # mp3_fp.seek(0)  # go to beginning
            # audio_segments.append((idx, AudioSegment.from_mp3(mp3_fp)))

            arr = np.array(s_t)
            audio_segment = AudioSegment(
                (arr * 32767 / max(0.01, np.max(np.abs(arr)))).astype(np.int16).tostring(),
                frame_rate=22050,
                channels=1,
                sample_width=2
            )
            audio_segments.append((idx, audio_segment))

        # a = AudioSegment(data=np.array(synthesized_texts[0]), sample_width=1, channels=1, frame_rate=44100)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(inner, idx, s_t) for idx, s_t in enumerate(synthesized_texts)]

        wait(futures)

        audio_segments = sorted(audio_segments, key=lambda x: x[0])
        audio_segments = [el[1] for el in audio_segments]

        return audio_segments

