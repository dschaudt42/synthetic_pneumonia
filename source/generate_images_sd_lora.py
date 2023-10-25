import os
import shutil
import re
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


model_path = "diffusers/SD_lora/1e-4"
output_dir1 = 'images/final/sd_lora/1e-4/sd_lora_scale05'
output_dir2 = 'images/final/sd_lora/1e-4/sd_lora_scale1'
num_images = 1000
num_steps = 50
best_iteration_scale05 = 14500
best_iteration_scale1 = 2500


class_prompt_map = {'V' : "An x-ray image of the lung with viral pneumonia",
                    'B' : "An x-ray image of the lung with bacterial pneumonia",
                    'C' : "An x-ray image of the lung with covid-19 pneumonia",
                    'F' : "An x-ray image of the lung with fungal pneumonia",
                    'H' : "An x-ray image of the lung, healthy patient, no signs of pneumonia"
                   }


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.bfloat16)
pipe.unet.load_attn_procs(f'{model_path}/checkpoint-{best_iteration_scale05}')
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

for label,prompt in class_prompt_map.items():
        os.makedirs(f'{output_dir1}/{label}',exist_ok=True)

        for i in range(num_images):
            image = pipe(prompt=prompt, num_inference_steps=num_steps, cross_attention_kwargs={"scale": 0.5}).images[0]
            image.save(f'{output_dir1}/{label}/image-{i}.png')

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.bfloat16)
pipe.unet.load_attn_procs(f'{model_path}/checkpoint-{best_iteration_scale1}')
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

for label,prompt in class_prompt_map.items():
        os.makedirs(f'{output_dir2}/{label}',exist_ok=True)

        for i in range(num_images):
            image = pipe(prompt=prompt, num_inference_steps=num_steps, cross_attention_kwargs={"scale": 1.0}).images[0]
            image.save(f'{output_dir2}/{label}/image-{i}.png')



model_path = "diffusers/SD_lora/1e-5"
output_dir1 = 'images/final/sd_lora/1e-5/sd_lora_scale05'
output_dir2 = 'images/final/sd_lora/1e-5/sd_lora_scale1'
best_iteration_scale05 = 15000
best_iteration_scale1 = 2000

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.bfloat16)
pipe.unet.load_attn_procs(f'{model_path}/checkpoint-{best_iteration_scale05}')
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

for label,prompt in class_prompt_map.items():
        os.makedirs(f'{output_dir1}/{label}',exist_ok=True)

        for i in range(num_images):
            image = pipe(prompt=prompt, num_inference_steps=num_steps, cross_attention_kwargs={"scale": 0.5}).images[0]
            image.save(f'{output_dir1}/{label}/image-{i}.png')

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.bfloat16)
pipe.unet.load_attn_procs(f'{model_path}/checkpoint-{best_iteration_scale1}')
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")

for label,prompt in class_prompt_map.items():
        os.makedirs(f'{output_dir2}/{label}',exist_ok=True)

        for i in range(num_images):
            image = pipe(prompt=prompt, num_inference_steps=num_steps, cross_attention_kwargs={"scale": 1.0}).images[0]
            image.save(f'{output_dir2}/{label}/image-{i}.png')