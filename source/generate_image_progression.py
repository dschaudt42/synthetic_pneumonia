import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import glob
import diffusers
from typing import List, Optional, Tuple, Union
from diffusers import DDPMPipeline, DDPMScheduler 
from PIL import Image


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")


    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


device = "cuda"
generator = torch.Generator("cuda").manual_seed(1024)

output = 'export/img_progression'

num_steps = 10


pipeline = DDPMPipeline.from_pretrained(f'diffusers/H_190/epoch-189', local_files_only=True)
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
pipeline.to(device)


pipeline.scheduler.set_timesteps(num_steps)

if isinstance(pipeline.unet.sample_size, int):
    image_shape = (1, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)
else:
    image_shape = (1, pipeline.unet.in_channels, *pipeline.unet.sample_size)



image = randn_tensor(image_shape, generator=generator, device=device)


for i,t in enumerate(pipeline.progress_bar(pipeline.scheduler.timesteps)):
    # 1. predict noise model_output
    model_output = pipeline.unet(image, t).sample

    # 2. compute previous image: x_t -> x_t-1
    image = pipeline.scheduler.step(model_output, t, image, generator=generator).prev_sample

    image_output = (image / 2 + 0.5).clamp(0, 1)
    image_output = image_output.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
    #image_output = image_output.detach().cpu().numpy().squeeze()

    image_output = (image_output * 255).round().astype("uint8")
    im = Image.fromarray(image_output,mode='RGB')
    im.save(f'{output}/image-{i}.png')
