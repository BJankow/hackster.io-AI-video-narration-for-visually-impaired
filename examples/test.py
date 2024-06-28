import torch

print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time

device = "cuda:0"
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=False)
model.to(device)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
print(f"{model.num_parameters() / 1e9:.2f}B parameters")
print(model)


def run_inference(text='', image_source='', is_url=False):
    # load image from either a url or local file
    if is_url == True:
        image = Image.open(requests.get(image_source, stream=True).raw)
    else:
        image = Image.open(image_source)
        image = image.convert("RGB")
    # show image
    image.show()

    # create prompt and process input
    start = time.time()
    prompt = f"<image>\nUSER: {text}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate response
    generate_ids = model.generate(**inputs, max_length=500)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Generated in {time.time() - start: .2f} secs")
    print(response)
    return response


text = "Identify the landmark in the image and tell me one fun fact about it"
response = run_inference(text, "https://rocm.blogs.amd.com/_images/example6.jpg", is_url=True)
