import numpy as np
import pandas as pd
import math
import glob
from pathlib import Path
import torch
import torchmetrics
import itertools
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from PIL import Image
import matplotlib.pyplot as plt

def load_img(input):
    img = Image.open(input).resize((299,299))
    img = np.transpose(img,(2,0,1))
    return img


def read_imgs_to_tensor(file_path_list):
    img_array = np.array([np.array(load_img(fname)) for fname in file_path_list])
    img_array = torch.from_numpy(img_array)
    return img_array


def calc_fid(real_img_array,gen_image_array):
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_img_array, real=True)
    fid.update(gen_image_array, real=False)
    return fid.compute()

def calc_pairwise_msssim_values(img_array):
    ms_ssim_values = []
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=255)
    for x,y in list(itertools.combinations(range(img_array.shape[0]), r=2)):
        ms_ssim_values.append(ms_ssim(img_array[x].unsqueeze(0).to(torch.float32), img_array[y].unsqueeze(0).to(torch.float32)).item())
    return ms_ssim_values


def create_df(epoch_list,fid_list,ms_ssim_list,train_image_folder):
    df = pd.DataFrame(data={'epoch':epoch_list,'fid':fid_list,'ms_ssim':ms_ssim_list})
    df['epoch'] = df['epoch'].astype('int')
    df.sort_values('epoch',inplace=True)

    num_train_imgs = len(glob.glob(f'{train_image_folder}/*'))
    iterations_per_epoch = math.ceil(num_train_imgs/8)
    df['iterations'] = (df['epoch']+1)*iterations_per_epoch
    return df


models = ['B_1250','C_1140','H_190','P_1000','V_1700']
train_folders = ['B','C','NB','P','V']

for model,train_folder in tqdm(zip(models,train_folders)):

    train_image_folder = f'../../data/segmentation_test/train_per_class/{train_folder}'
    train_image_files = glob.glob(f'{train_image_folder}/*')[0:50]

    epoch_list = []
    fid_list = []
    ms_ssim_list = []

    for folder in tqdm(glob.glob(f'images/unconditional/diffusers/{model}/*')):
        epoch_num = Path(folder).name[6:]
        diffusion_image_files = glob.glob(f'{folder}/*')

        diff_images = read_imgs_to_tensor(diffusion_image_files)
        real_images = read_imgs_to_tensor(train_image_files)

        fid = calc_fid(real_images,diff_images)

        ms_ssim_values = calc_pairwise_msssim_values(diff_images)
        mean_ms_ssim = np.mean(ms_ssim_values)

        epoch_list.append(epoch_num)
        fid_list.append(fid.item())
        ms_ssim_list.append(mean_ms_ssim)

    df = create_df(epoch_list,fid_list,ms_ssim_list,train_image_folder)
    df.to_csv(f'export/train_metrics/unconditional/{model}_train_metrics.csv',index=False)
