# standard library imports
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Union, Optional, Tuple
from tqdm import tqdm

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub.audio_segment import AudioSegment
from pydub.playback import play
from scenedetect import FrameTimecode, VideoStreamCv2

# local imports
from .MovieComposerInterface import MovieComposerInterface
from utils.LogHandling.LogHandlers import StandardLogger


class MovieComposerBase(MovieComposerInterface, StandardLogger):

    def __init__(self):
        super(MovieComposerBase, self).__init__()
        # Temporary files that are deleted when closed or on program termination.
        self.__video_tmp_fp: Optional[Union[str, Path]] = None  # path to optional temporary file with video
        self.__audio_tmp_fp: Optional[Union[str, Path]] = None  # path to optional temporary file with audio

    def compose(
            self,
            video_fp: Union[str, Path],
            audio_fp: Union[str, Path],
            scenes: List[Tuple[FrameTimecode, FrameTimecode]],
            synthesized_descriptions: List[AudioSegment]
    ):
        """
        Composes Video and Audio files into movie.
        :param video_fp: filepath to video file.
        :param audio_fp: filepath to audio file.
        :param scenes: scenes as Tuple of FrameTimecodes. First indicates beginning of the scene, second - end.
        :param synthesized_descriptions: descriptions as audio.
        :return:
        """
        video = VideoFileClip(video_fp)
        audio = AudioSegment.from_file(audio_fp)

        video_frame_rate = video.fps  # [FPS]
        audio_frequency = audio.frame_rate  # [Hz]
        audio_duration = audio.duration_seconds  # [s]
        audio_frame_rate = len(audio) / audio_duration  # [FPS]

        freeze_command = f"ffmpeg -an -y -i {video_fp}"
        freeze_command_sequence = "PTS-STARTPTS"

        new_audio = AudioSegment.empty()
        for scene, synthesized in tqdm(zip(scenes, synthesized_descriptions), desc="Adding synthesized audio into movie..."):
            synthesized_duration = synthesized.duration_seconds  # [s]
            scene_start_frame = scene[0].frame_num
            scene_stop_frame = scene[1].frame_num
            freeze_command_sequence += f" + gte(T,{(scene_start_frame + 1) / video_frame_rate})*({synthesized_duration}/TB)"

            # Modify Audio
            synthesized = synthesized.set_frame_rate(audio_frequency) - 2
            start_pos = int(scene_start_frame * audio_frame_rate / video_frame_rate)
            stop_pos = int(scene_stop_frame * audio_frame_rate / video_frame_rate)
            new_audio += synthesized
            new_audio += audio[start_pos: stop_pos]

        freeze_command += f" -vf \"setpts='{freeze_command_sequence}'\""  #  -af \"asetpts='{freeze_command_sequence}',aresample=async=1:first_pts=0\"

        self.__audio_tmp_fp = NamedTemporaryFile(suffix=".mp3")
        new_audio.export(out_f=self.__audio_tmp_fp.name)

        # Modify Movie
        self.__video_tmp_fp = NamedTemporaryFile(suffix=".mov")
        freeze_command += f" -vsync cfr {self.__video_tmp_fp.name}"
        os.system(freeze_command)

    def save(self, out_fp: Union[str, Path]):
        """
        Saves composed movie as a file.

        :param out_fp: path to output file.
        :return:
        """
        assert self.__video_tmp_fp is not None
        assert os.path.isfile(self.__video_tmp_fp.name)
        assert self.__audio_tmp_fp is not None
        assert os.path.isfile(self.__audio_tmp_fp.name)

        os.system(f"ffmpeg -y -i {self.__video_tmp_fp.name} -i {self.__audio_tmp_fp.name} -c:v copy -c:a copy {out_fp}")