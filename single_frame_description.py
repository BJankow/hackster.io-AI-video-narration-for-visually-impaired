# standard library imports
import os

# 3rd party library imports
import cv2

# local imports
from StagesProcessor import MovieHandlerBase, ClipDescriptorLLaVA15

DATASETS_FOLDER = "../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset")
OUTPUT_DIR = "single_frame_description_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILENAME = "big_buck_bunny_1080p_h264.mov"
VID_PATH = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)


if __name__ == '__main__':
    movie_handler = MovieHandlerBase()
    movie_handler.load(fp=VID_PATH)
    frame = movie_handler.get_frame(float(3*60 + 18))
    descriptor = ClipDescriptorLLaVA15()
    description = descriptor.describe_single_image(image=frame)

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'described_frame.png'), frame)

    # SAVE DESCRIPTION
    f = open(os.path.join(OUTPUT_DIR, 'description.txt'), 'w')
    f.write(description)
    f.close()
