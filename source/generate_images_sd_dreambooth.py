import os
import re
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


model_path = "diffusers/SD_dreambooth/text_encoder_frozen"
output_dir = 'images/final/sd_dreambooth'
num_images = 1000
num_steps = 50
best_model_iteration = {'B':300,'C':300,'H':300,'F':1500,'V':1500}

class_prompt_map = {'V' : "An x-ray image of the lung with viral pneumonia",
                    'B' : "An x-ray image of the lung with bacterial pneumonia",
                    'C' : "An x-ray image of the lung with covid-19 pneumonia",
                    'F' : "An x-ray image of the lung with fungal pneumonia",
                    'H' : "An x-ray image of the lung, healthy patient, no signs of pneumonia"
                   }


for label,prompt in tqdm(class_prompt_map.items()):

    unet = UNet2DConditionModel.from_pretrained(f'{model_path}/{label}/checkpoint-{best_model_iteration[label]}/unet')

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet)#, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    os.makedirs(f'{output_dir}/{label}',exist_ok=True)

    for i in range(num_images):
            image = pipe(prompt=prompt,num_inference_steps=num_steps).images[0]
            image.save(f'{output_dir}/{label}/image-{i}.png')

