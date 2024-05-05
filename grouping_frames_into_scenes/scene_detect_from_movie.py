# standard library imports
import logging
import math
import os
import pickle
import sys
from typing import List

# 3rd party library imports
import cv2
from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector, split_video_ffmpeg

# local imports


DATASETS_FOLDER = "../../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset/")
FILENAME = "big_buck_bunny_1080p_h264.mov"
FILEPATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)
CACHE_FOLDER = "scene_detect_cache"
PICKLE_FILE = os.path.join(CACHE_FOLDER, os.path.splitext(FILENAME)[0] + ".pickle")

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s) |%(name)s|: %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    if os.path.exists(PICKLE_FILE):
        scenelist = pickle.load(open(PICKLE_FILE, 'rb'))
    else:
        scenelist = detect(
            FILEPATH,
            AdaptiveDetector(),
            show_progress=True  # show progress of video processing
        )
        pickle.dump(scenelist, open(PICKLE_FILE, 'wb'))

    end_scene_frame_indices: List[int] = [el[1].frame_num for el in scenelist]

    next_end_scene_idx = end_scene_frame_indices.pop(0)  # remove and return 1st element from the list - 1st cut

    # Animate till cuts
    frame_idx = 0
    cap = cv2.VideoCapture(FILEPATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_threshold_length = math.ceil(math.log10(total_frames))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:  # end of clip
            break

        while frame_idx == next_end_scene_idx:  # STOP SCENE, wait for 'k' pressed
            if cv2.waitKey(0) & 0xFF == ord('k'):
                next_end_scene_idx = end_scene_frame_indices.pop(0)  # obtain and remove subsequent cuts
                break

        cv2.imshow(FILENAME, frame)
        logging.debug(f"Showing frame idx: {str(frame_idx + 1).zfill(total_frames_threshold_length)}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    pass
