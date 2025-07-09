# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/test-sd-15/model_cache/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import mimetypes

mimetypes.add_type("image/webp", ".webp")

import time
import torch
import subprocess
from typing import Optional
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        model_files = [
            "models--sd-legacy--stable-diffusion-v1-5.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Load the model
        model_id = "sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, cache_dir=MODEL_CACHE
        )
        self.pipe = self.pipe.to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale for text conditioning",
            ge=1.0,
            le=20.0,
            default=7.5,
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Leave blank for a random seed.",
            default=None,
        ),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="The image compression quality (for lossy formats like JPEG and WebP). 100 = best quality, 0 = lowest quality.",
            ge=1,
            le=100,
            default=80,
        ),
    ) -> Path:
        """Generate an image from a text prompt"""
        # Set up generator with seed if provided
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Generate image
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        ).images[0]

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prepare saving arguments
        extension = output_format.lower()
        save_params = {}

        # Add quality parameter for lossy formats
        if output_format != "png":
            print(f"[~] Output quality: {output_quality}")
            save_params["quality"] = output_quality
            save_params["optimize"] = True

        # Handle jpg/jpeg naming
        if extension == "jpg":
            extension = "jpeg"

        # Create output path
        output_path = Path(f"output.{extension}")

        # Save the image with appropriate parameters
        image.save(str(output_path), **save_params)
        print(f"[+] Saved output as {output_format.upper()}")

        return output_path
