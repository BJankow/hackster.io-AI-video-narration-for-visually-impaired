# standard library imports
from io import BytesIO
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Union, Optional, Tuple
from tqdm import tqdm

# 3rd party library imports
from moviepy.editor import VideoFileClip, AudioFileClip
from pyannote.audio.pipelines import VoiceActivityDetection
from pydub.audio_segment import AudioSegment
from pydub.effects import speedup
from pydub.playback import play
from scenedetect import FrameTimecode, VideoStreamCv2

# local imports
from .MovieComposerInterface import MovieComposerInterface
from utils.LogHandling.LogHandlers import StandardLogger


TIME_DISTANCE_BETWEEN_LECTOR_AND_EVENT = 1  # [s] - time between lector ending speaking and (end of scene or beginning of someone talking)


class MovieComposerBase(MovieComposerInterface, StandardLogger):

    def __init__(self):
        super(MovieComposerBase, self).__init__()
        # Temporary files that are deleted when closed or on program termination.
        self.__video_tmp_fp: Optional[Union[str, Path]] = None  # path to optional temporary file with video
        self.__audio_tmp_fp: Optional[Union[str, Path]] = None  # path to optional temporary file with audio
        self.__voice_detector: Optional[VoiceActivityDetection] = None

    def __detect_voice_fragments(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        """
        Detects moments in an audio segment where actual voice exists.
        :param audio: audio segment.
        :return: List of Tuples in following form: [(t0_start, t0_end), (t1_start, t1_end)..., (tn_start, tn_end)].
            Every tuple contains start and stop moment of speech detected. Unit: [s].
            Empty list is returned when no speech is detected.
        """

        if self.__voice_detector is None:
            self.__voice_detector = VoiceActivityDetection(segmentation="anilbs/segmentation")
            HYPER_PARAMETERS = {
                # onset/offset activation thresholds
                "onset": 0.95, "offset": 0.3,
                # remove speech regions shorter than that many seconds.
                "min_duration_on": 0.0,
                # fill non-speech regions shorter than that many seconds.
                "min_duration_off": 0.0
            }
            self.__voice_detector.instantiate(HYPER_PARAMETERS)

        buffer = BytesIO()
        audio.export(buffer, format="wav")
        vad = self.__voice_detector(buffer)

        if len(vad._tracks._list._lists) == 0:  # No Speeck detected:
            return []

        return [(el.start, el.end) for el in vad._tracks._list._lists[0]]

    def __detect_first_voice_fragment(self, audio: AudioSegment) -> Optional[Tuple[float, float]]:
        """
        Detects first appearance of speech in an audio segment.

        :param audio: audio segment.
        :return: Tuple in following form (t_start, t_stop) which describes start and stop moment of speech .Unit: [s].
            If no speech is detected then None is returned.
        """
        moments_of_speech = self.__detect_voice_fragments(audio=audio)
        if len(moments_of_speech) > 0:
            return moments_of_speech[0]

        return None

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

        freeze_command = f"ffmpeg -an -y -i \"{video_fp}\""
        freeze_command_sequence = "PTS-STARTPTS"

        new_audio = AudioSegment.empty()
        last_stop_pos = 0
        for scene, synthesized in tqdm(
                zip(scenes, synthesized_descriptions),
                desc=f"Adding synthesized audio into movie... (total: {len(synthesized_descriptions)})"
        ):
            synthesized = synthesized.set_frame_rate(audio_frequency) + 3

            synthesized_duration = synthesized.duration_seconds  # [s]
            scene_start_frame = scene[0].frame_num
            scene_stop_frame = scene[1].frame_num
            scene_duration = (scene_stop_frame - scene_start_frame) / video_frame_rate

            start_pos = int(scene_start_frame * audio_frame_rate / video_frame_rate)
            stop_pos = int(scene_stop_frame * audio_frame_rate / video_frame_rate)

            # Voice detection
            voice_details = self.__detect_first_voice_fragment(audio=audio[start_pos:stop_pos])
            if voice_details is None:
                synthesized_stop_t = scene_duration - TIME_DISTANCE_BETWEEN_LECTOR_AND_EVENT
            else:
                synthesized_stop_t = voice_details[1] - TIME_DISTANCE_BETWEEN_LECTOR_AND_EVENT

            # ADD LECTOR. Add lector at begin or earlier. Earlier - freeze some frames.
            synthesized_start_t = min(0, synthesized_stop_t - synthesized_duration)

            if synthesized_start_t < 0:
                freeze_command_sequence += f" + gte(T,{(scene_start_frame + 1) / video_frame_rate})*({-synthesized_start_t}/TB)"
                synthesized_before_0 = synthesized[: - synthesized_start_t * audio_frame_rate]
                synthesized_after_0 = synthesized[- synthesized_start_t * audio_frame_rate:]

                new_audio += synthesized_before_0  # append
                new_audio += audio[last_stop_pos: stop_pos].overlay(synthesized_after_0)  # merge
            else:  # synthesized_start_t at the beginning of scene.
                new_audio += audio[last_stop_pos: stop_pos].overlay(synthesized)

            last_stop_pos = stop_pos
        else:
            new_audio += audio[last_stop_pos: len(audio)]
            pass

        # TODO: adding synthesized makes missing few audio frames...
        # assert len(audio) + sum([len(s) for s in synthesized_descriptions]) == len(new_audio), \
        #     "You have missed some audio frames"  # no audio frames missed check
        freeze_command += f" -vf \"setpts='{freeze_command_sequence}'\""  #  -af \"asetpts='{freeze_command_sequence}',aresample=async=1:first_pts=0\"

        self.__audio_tmp_fp = NamedTemporaryFile(suffix=".mp3")
        new_audio.export(out_f=self.__audio_tmp_fp.name)

        # Modify Movie
        self.__video_tmp_fp = NamedTemporaryFile(suffix=".mov")
        freeze_command += f" -vsync cfr \"{self.__video_tmp_fp.name}\""
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

        os.system(f"ffmpeg -y -i \"{self.__video_tmp_fp.name}\" -i \"{self.__audio_tmp_fp.name}\" -c:v copy -c:a copy \"{out_fp}\"")