"""
Encode each frame from video and try to clusterize frames. Basing on that create chapters.
"""

# standard library imports
from datetime import datetime
import logging
import os.path
import sys

# 3rd party library imports
import cv2
import numpy as np
from PIL import Image
import torch
from torchsummary import summary
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from torchvision import transforms

# local imports

DATASETS_FOLDER = "../../datasets/"
CURRENT_DATASET_FOLDER = os.path.join(DATASETS_FOLDER, "hackster.io-AI-video-narration-for-visually-impaired-dataset/")
FILENAME = "big_buck_bunny_1080p_h264.mov"

PARALLEL_PROCESSED = 1500
VISUALIZE_EVERY_NTH_FRAME = 100

# Based on: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s) |%(name)s|: %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    movie_location = os.path.join(CURRENT_DATASET_FOLDER, FILENAME)

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    logging.debug(f"GPU DEVICE USED: {torch.cuda.get_device_name(0)}")

    # LOADING MODELS
    logging.info("Loading models...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = model.to(gpu_device)
    model.eval()

    logging.info(f"Encoder params: {'{:,}'.format(model.encoder.num_parameters())}, "
                 f"Decoder params: {'{:,}'.format(model.decoder.num_parameters())}")

    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")  # resize + normalize

    logging.info("Models loaded!")

    logging.debug(model)

    cap = cv2.VideoCapture(movie_location)

    images = []
    all_encoded = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_done = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frames_done += 1

        if not ret:
            break

        if frames_done % VISUALIZE_EVERY_NTH_FRAME == 0:
            cv2.imshow(FILENAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # images.append(Image.fromarray(frame))
        images.append(torch.from_numpy(frame))

        if frames_done % PARALLEL_PROCESSED == 0:  # after collecting amount=PARALLEL_PROCESSED images
            logging.debug(f"[{frames_done} / {total_frames}]")
            t = datetime.now()
            pixel_values = image_processor(torch.stack(images, dim=0), return_tensors="pt").pixel_values
            logging.debug(f"Image processor {PARALLEL_PROCESSED} images time: {(datetime.now() - t).total_seconds()} [s]")

            t = datetime.now()
            pixel_values = pixel_values.to(gpu_device)
            with torch.no_grad():
                encoded = model.encoder(pixel_values).pooler_output.to(cpu_device)
            logging.debug(f"Image Encoder {PARALLEL_PROCESSED} images time: {(datetime.now() - t).total_seconds()} [s]")
            all_encoded += encoded

            # getting ready for next images - cleaning memory
            images = []

    if len(images) > 0:
        pixel_values = image_processor(torch.stack(images, dim=0), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(gpu_device)
        with torch.no_grad():
            encoded = model.encoder(pixel_values).pooler_output.to(cpu_device)
        all_encoded += encoded
        del images

    cv2.destroyAllWindows()
    torch.cuda.empty_cache()  # free cache space in G-RAM

    all_encoded = torch.stack(all_encoded, dim=0)

    all_encoded.numpy()  # convert to numpy array
    np.save(os.path.splitext(FILENAME)[0] + "_encoded-frames.npy", all_encoded)
