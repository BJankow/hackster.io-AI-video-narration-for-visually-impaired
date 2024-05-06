# standard library imports
from typing import List, Tuple

# 3rd party library imports
from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector, split_video_ffmpeg, SceneManager, \
    VideoStream, FrameTimecode

# local imports
from .SceneDetectorInterface import SceneDetectorInterface


class SceneDetectorBase(SceneDetectorInterface):

    def __init__(self):
        super(SceneDetectorBase, self).__init__()
        self.__number_of_frames = None
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(AdaptiveDetector())

    def detect_scenes(self, video) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        self.__number_of_frames = self.scene_manager.detect_scenes(
            video=video,
            show_progress=True
        )
        scenes = self.scene_manager.get_scene_list()
        return scenes


