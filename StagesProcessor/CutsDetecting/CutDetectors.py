# standard library imports
from typing import List

# 3rd party library imports
from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector, split_video_ffmpeg, SceneManager, \
    VideoStream

# local imports
from .CutDetectorInterface import CutDetectorInterface


class CutDetectorBase(CutDetectorInterface):

    def __init__(self):
        super(CutDetectorBase, self).__init__()
        self.__number_of_frames = None
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(AdaptiveDetector())

    def detect_scenes(self, video) -> List:

        self.__number_of_frames = self.scene_manager.detect_scenes(
            video=video,
            show_progress=True
        )
        scenelist = self.scene_manager.get_scene_list()
        return scenelist


