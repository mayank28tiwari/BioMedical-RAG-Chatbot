from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image
import os
from datetime import datetime

class BiomedicalImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            revision="fp16" if self.device == "cuda" else None
        )
        self.pipe = self.pipe.to(self.device)
        os.makedirs("generated_images", exist_ok=True)
        print(f"Model loaded on {self.device}.")

    def format_prompt(self, query: str, additional: str = "") -> str:
        base = f"Create a Biomedical illustration of {query}"
        if additional:
            base += f",Strictly focusing on {additional}"
        return f"{base}, high quality, labeled, scientific visualization, medical diagram"

    def generate_image(self, query: str, seed: int = None, additional: str = "") -> Image.Image:
        prompt = self.format_prompt(query, additional)
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None
        result = self.pipe(prompt, num_inference_steps=30, generator=generator, guidance_scale=7.5)
        return result.images[0]

    def edit_image(self, original_query: str, edit_prompt: str) -> Image.Image:
        return self.generate_image(original_query, additional=edit_prompt)

    def save_image(self, image: Image.Image, filename: str = None) -> str:
        if not filename:
            filename = f"biomed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join("generated_images", filename)
        image.save(path)
        return path

