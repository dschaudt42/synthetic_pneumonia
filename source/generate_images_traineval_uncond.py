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

output = 'images/unconditional'

models = ['B_1250','C_1140','H_190','P_1000','V_1700']
num_images = 50
num_steps = 250

for model in models:
    basedir = f'diffusers/{model}'
    model_list = glob.glob(f'{basedir}/epoch-*')

    for model_save in model_list:
        pipeline = DDPMPipeline.from_pretrained(f'{model_save}', local_files_only=True)
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(device)

        os.makedirs(f'{output}/{model_save}',exist_ok=True)

        for i in range(num_images):
            # generate image
            image = pipeline(num_inference_steps=num_steps).images[0]
            # save image
            image.save(f"{output}/{model_save}/image-{i}.png")
