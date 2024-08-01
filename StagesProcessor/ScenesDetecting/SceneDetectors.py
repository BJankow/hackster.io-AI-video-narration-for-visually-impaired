# standard library imports
from typing import List, Tuple

import numpy as np
# 3rd party library imports
from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector, split_video_ffmpeg, SceneManager, \
    VideoStream, FrameTimecode, VideoStreamCv2

# local imports
from .SceneDetectorInterface import SceneDetectorInterface
from utils.LogHandling.LogHandlers import StandardLogger


class SceneDetectorBase(SceneDetectorInterface, StandardLogger):

    def __init__(self):
        super(SceneDetectorBase, self).__init__()
        self.__number_of_frames = None
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(AdaptiveDetector())

    def detect_scenes(
            self,
            video: VideoStreamCv2,
            time_stop: float,
            time_start: float
    ) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        Detects scenes in a given serie of frames.

        :param video: video to be processed.
        :param time_stop: (seconds) in what moment of the movie to stop processing.
        :param time_start: (seconds) in what moment of the movie to start processing.
        :return:
        """
        if time_stop < np.inf:
            end_time = FrameTimecode(time_stop, fps=video.frame_rate)
        else:
            end_time = None

        self.__number_of_frames = self.scene_manager.detect_scenes(
            video=video,
            show_progress=True,
            end_time=end_time
        )
        scenes = self.scene_manager.get_scene_list()

        if time_start > 0.0:
            scenes = [s for s in scenes if s[0].get_seconds() < time_start and s[1].get_seconds() > time_stop]

        return scenes


