'''
RunPod Serverless handler for Pony Diffusion V6 XL.
Optimized for FORGE anime avatar pipeline — generates NSFW anime character portraits.
'''

import os
import base64
import concurrent.futures

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


# ------------------------------- Model Handler ------------------------------ #

MODEL_URL = "https://huggingface.co/AstraliteHeart/pony-diffusion-v6/blob/main/v6.safetensors"


class ModelHandler:
    def __init__(self):
        self.base = None
        self.load_models()

    def load_base(self):
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_URL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe = pipe.to("cuda", silence_dtype_warnings=True)
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            self.base = future_base.result()


MODELS = ModelHandler()


# ---------------------------------- Helper ---------------------------------- #

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_A": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    '''Generate anime character portraits via Pony Diffusion V6 XL.'''
    job_input = job["input"]

    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    MODELS.base.scheduler = make_scheduler(
        job_input['scheduler'], MODELS.base.scheduler.config)

    starting_image = job_input.get('image_url')

    if starting_image:
        from diffusers.utils import load_image
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            num_inference_steps=job_input['num_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator,
        ).images
    else:
        try:
            output = MODELS.base(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                height=job_input['height'],
                width=job_input['width'],
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                num_images_per_prompt=job_input['num_images'],
                generator=generator,
            ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}",
                "refresh_worker": True,
            }

    image_urls = _save_and_upload_images(output, job['id'])

    return {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed'],
    }


runpod.serverless.start({"handler": generate_image})
