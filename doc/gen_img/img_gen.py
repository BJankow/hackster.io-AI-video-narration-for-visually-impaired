# standard library imports

# 3rd party library imports
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# local imports


if __name__ == '__main__':
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to('cuda:0')

    # prompt = "Happy confident narrator talking in boxy TV animated style"
    prompt = "Generate a portrait image showcasing a person’s expressions, features, and emotions in a close-up shot. Keep the background neutral, emphasizing the individual’s presence. Bring out the depth and personality of the subject, creating a captivating and well-composed portrait."

    pipe.enable_attention_slicing()

    for i in range(10):
        image = pipe(prompt).images[0]
        image.save(f"generated_{i}.png")

