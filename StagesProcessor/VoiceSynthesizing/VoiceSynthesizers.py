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

READERS_FOLDER = os.path.join('../../voice_samples/')
LANGUAGE2READER = {
    'en': os.path.join(READERS_FOLDER, 'en.wav'),
    'pl': os.path.join(READERS_FOLDER, 'pl.wav')
}


class VoiceSynthesizerBase(VoiceSynthesizerInterface):

    def __init__(self):
        super(VoiceSynthesizerBase, self).__init__()
        # self.__engine = pyttsx3.init()

    def synthesize(self, texts: List[str], language: str) -> List[AudioSegment]:

        # synthesized_texts = [gTTS(text=text, lang='en', slow=False) for text in texts]
        if language not in LANGUAGE2READER.keys():
            raise ValueError(f"Given language ({language}) is not supported. "
                             f"Supported languages: {LANGUAGE2READER.keys()}")

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True, progress_bar=True)
        tts.eval()

        synthesized_text_files = [BytesIO() for _ in texts]

        # You can check all Coqui available speakers with the following command:
        # tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --list_speaker_idx
        for f, text in zip(synthesized_text_files, texts):
            tts.tts_to_file(
                text=text,
                file_path=f,
                speaker_wav=LANGUAGE2READER[language],
                language=language,
                temperature=0.3,
                repetition_penalty=25.0,  # Default: 2.0
                length_penalty=30,  # Default: 1.0
                top_k=1,  # Default: 50
                top_p=0.0,  # Default: 0.8
                split_sentences=False,
                enable_text_splitting=False  # Default: True
            )
        audio_segments = []

        def inner(idx: int, file: BytesIO):
            audio_segment = AudioSegment.from_file(file)
            audio_segments.append((idx, audio_segment))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(inner, idx, f) for idx, f in enumerate(synthesized_text_files)]

        wait(futures)

        audio_segments = sorted(audio_segments, key=lambda x: x[0])
        audio_segments = [el[1] for el in audio_segments]

        return audio_segments

