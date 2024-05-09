# standard library imports
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List
from tqdm import tqdm

# 3rd party library imports
from gtts import gTTS
from pydub.audio_segment import AudioSegment
from pydub.silence import detect_silence

import pyttsx3

# local imports
from .VoiceSynthesizerInterface import VoiceSynthesizerInterface


class VoiceSynthesizerBase(VoiceSynthesizerInterface):

    def __init__(self):
        super(VoiceSynthesizerBase, self).__init__()
        # self.__engine = pyttsx3.init()

    def synthesize(self, texts: List[str]) -> List[AudioSegment]:

        synthesized_texts = [gTTS(text=text, lang='en', slow=False) for text in texts]
        audio_segments = []
        total_amount = len(texts)

        def inner(idx, s_t):
            mp3_fp = BytesIO()  # write to buffer
            s_t.write_to_fp(mp3_fp)
            mp3_fp.seek(0)  # go to beginning
            audio_segments.append((idx, AudioSegment.from_mp3(mp3_fp)))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(inner, idx, s_t) for idx, s_t in enumerate(synthesized_texts)]

        wait(futures)

        audio_segments = sorted(audio_segments, key=lambda x: x[0])
        audio_segments = [el[1] for el in audio_segments]

        return audio_segments

