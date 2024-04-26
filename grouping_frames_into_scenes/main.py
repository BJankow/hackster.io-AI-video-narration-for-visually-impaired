# standard library imports
import os.path

# 3rd party library imports
import cv2
import numpy as np
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# local imports

DATASETS_FOLDER = "../../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset/")


# Based on: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
if __name__ == '__main__':
    movie_location = os.path.join(CURRENT_DATASET_FOLDER, "big_buck_bunny_1080p_h264.mov")

    print("Loading models...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    print("Models loaded!")

    cap = cv2.VideoCapture(movie_location)
    idx = -1
    while cap.isOpened():
        ret, frame = cap.read()
        idx += 1
        cv2.imshow('FRAME', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        image = Image.fromarray(frame)

        if idx == 100:
            pixel_values = image_processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_text)

    # Encode each frame from video and try to clusterize frames. Basing on that create chapters.
