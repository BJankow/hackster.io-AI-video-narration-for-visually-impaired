# standard library imports
from io import BytesIO, FileIO
import os
from pathlib import Path
from pydub.playback import play

# 3rd party library imports
import numpy as np
from pydub.audio_segment import AudioSegment
import torch
from TTS.api import TTS

# local imports

READERS_FOLDER = os.path.join(Path.home(), 'Projects', 'datasets', 'hackster.io-AI-video-narration-for-visually-impaired-lectors')
LANGUAGE2READER = {
    'en': os.path.join(READERS_FOLDER, 'AlanWatts.mp3'),
    # 'pl': os.path.join(READERS_FOLDER, 'Knapik.mp3'),
    'pl': os.path.join(READERS_FOLDER, 'DariuszSzpakowski.mp3')
    # 'pl': os.path.join(READERS_FOLDER, 'MagdalenaSchejbal.mp3')
}

LANGUAGE = 'pl'
if __name__ == '__main__':
    text = "Szukając trawy w buszu znalazłem światło na końcu tunelu."
    # text = "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device=device)
    tts.eval()

    f = BytesIO()
    tts.tts_to_file(
        text=text,
        file_path=f,
        # speaker="Sofia Hellen",
        speaker_wav=LANGUAGE2READER[LANGUAGE],
        language=LANGUAGE,
        temperature=0.3,  # Default: 0.65
        repetition_penalty=10.0,  # Default: 2.0
        length_penalty=30,  # Default: 1.0
        top_k=1,  # Default: 50
        top_p=0.0,  # Default: 0.8
        enable_text_splitting=False  # Default: True
    )
    b = AudioSegment.from_file(f)
    play(b)

    # arr = np.array(tts.tts(
    #     text=text,
    #     speaker="Sofia Hellen",
    #     language=LANGUAGE,
    #     temperature=2.0,  # Default: 0.65
    #     repetition_penalty=10.0,  # Default: 2.0
    #     length_penalty=30,  # Default: 1.0
    #     top_k=1,  # Default: 50
    #     top_p=0.0,  # Default: 0.8
    #     enable_text_splitting=False  # Default: True
    # ))
    # a = AudioSegment(
    #     (arr * 32767 / max(0.01, np.max(np.abs(arr)))).astype(np.int16).tostring(),
    #     frame_rate=22050,
    #     channels=1,
    #     sample_width=2,
    # )
    # play(a)

