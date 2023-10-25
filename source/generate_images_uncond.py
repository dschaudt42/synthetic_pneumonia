import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import glob
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler 

device = "cuda"

output = 'images/final/unconditional'

best_model_epoch = {'B_1250':750,'C_1140':616,'H_190':159,'P_1000':520,'V_1700':1699}
num_images = 1000
num_steps = 250

for model,epoch in tqdm(best_model_epoch.items()):
    model_dir = f'diffusers/{model}/epoch-{epoch}'

    pipeline = DDPMPipeline.from_pretrained(f'{model_dir}', local_files_only=True)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    os.makedirs(f'{output}/{model[:1]}',exist_ok=True)

    for i in range(num_images):
        # generate image
        image = pipeline(num_inference_steps=num_steps).images[0]
        # save image
        image.save(f"{output}/{model[:1]}/image-{i}.png")

os.rename('images/final/unconditional/P','images/final/unconditional/F')