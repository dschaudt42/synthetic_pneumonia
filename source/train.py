import os
import sys
import configparser
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_object_to_str(obj):
    if isinstance(obj, Path):
        return str(obj) 

def parse_config():
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    return config

def create_additional_data_frame(folder,num_images=1000):
    files = [item for row in [glob.glob(f'{folder}/{x}/*')[:num_images] for x in ['B','C','V','H','F']] for item in row]
    file_paths = [Path(x).resolve() for x in files]
    file_names = [Path(x).name for x in file_paths]
    classes = [Path(x).parent.name for x in file_paths]

    df_synth = pd.DataFrame(zip(file_paths,file_names,classes),columns=['file_path','file_name','class'])

    df_synth['file_path'] = df_synth['file_path'].apply(lambda x: path_object_to_str(x))
    df_synth['file_name'] = df_synth['file_name'].apply(lambda x: path_object_to_str(x))
    df_synth['split_old'] = 'train'
    df_synth['split'] = 'train'
    df_synth['patient_id'] = 'synth'

    return df_synth


def create_final_dataframe(metadata_file,synth_df):
    df = pd.concat([pd.read_csv(metadata_file),synth_df],ignore_index=True)

    # encode labels
    df['class'] = df['class'].astype('category')
    df['label_encoded'] = df['class'].cat.codes.astype('int64')

    return df


def get_weak_transforms(img_size,img_mean,img_std):
    weak_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return weak_transforms

def get_strong_transforms(img_size,img_mean,img_std):
    strong_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
                        A.OneOf(
                            [
                                A.CLAHE(p=1),
                                A.RandomBrightnessContrast(p=1),
                                A.RandomGamma(p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.Sharpen(p=1),
                                A.Blur(blur_limit=3, p=1),
                                A.MotionBlur(blur_limit=3, p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.RandomBrightnessContrast(p=1),
                                A.HueSaturationValue(p=1),
                            ],
                            p=0.9,
                        ),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return strong_transforms

def get_valid_transforms(img_size,img_mean,img_std):
    valid_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ])
    return valid_transforms


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations."""
    
    def __init__(
            self,
            df,
            augmentation=None,
            visualize = False
    ):
        self.df = df.reset_index(drop=True)
        self.ids = self.df.loc[:,'file_name'].values
        self.images_fps = self.df.loc[:,'file_path'].values
        
        self.augmentation = augmentation
        self.visualize = visualize
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.loc[i,'label_encoded']
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        
        # Revert Normalize to visualize the image
            if self.visualize:
                invTrans = A.Normalize(mean=[-x/y for x,y in zip(img_mean,img_std)],
                                       std=[1/x for x in img_std],
                                       max_pixel_value=1.0,
                                       always_apply=True)
                image = image.detach().cpu().numpy().transpose(1,2,0)
                image = invTrans(image=image)['image']
                image = (image*255).astype(np.uint8)
        
        return image, label
        
    def __len__(self):
        return len(self.ids)


def load_model(model_architecture,dropout,num_classes,pretrained=True):
    if model_architecture == 'convnext_small' or model_architecture == 'convnext_tiny':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes,drop_rate=dropout)

    if model_architecture == 'efficientnet_b0' or model_architecture == 'efficientnet_b1':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes)
        num_ftrs = model.get_classifier().in_features
        if dropout:
            model.classifier = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(num_ftrs,num_classes)
            )
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)

    if model_architecture == 'resnet50':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes)
        num_ftrs = model.get_classifier().in_features
        if dropout:
            model.fc = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(num_ftrs,num_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def main(config):
    # Configuration
    # Data
    print(f"Starting Experiment: {config['settings']['experiment_name']}")
    print('--------------------------------')
    
    img_mean = IMAGENET_DEFAULT_MEAN
    img_std = IMAGENET_DEFAULT_STD

    # Loop over additional synthetic images
    for num_images in eval(config['settings']['num_synth_images']):
        print(f'Start Training with {num_images} additional images')
        print('--------------------------------')

        df_synth = create_additional_data_frame(config['settings']['additional_data_folder'],num_images=num_images)
        df = create_final_dataframe(config['settings']['metadata_file'],df_synth)

        # Training and model
        device = torch.device(f"cuda:{config['settings']['cuda']}" if torch.cuda.is_available() else "cpu")
        num_classes = 5

        # Init data
        if config['model']['augmentations'] == 'strong':
            augmentations = get_strong_transforms(int(config['model']['resolution']),img_mean,img_std)
        elif config['model']['augmentations'] == 'weak':
            augmentations = get_weak_transforms(int(config['model']['resolution']),img_mean,img_std)

        train_dataset = Dataset(
            df[df['split']=='train'],
            augmentation=augmentations, 
        )

        valid_dataset = Dataset(
            df[df['split']=='valid'],
            augmentation=get_valid_transforms(int(config['model']['resolution']),img_mean,img_std), 
        )

        run_train_acc = {}
        run_val_acc = {}
        run_train_loss = {}
        run_val_loss = {}

        # Loop over repeated experiments
        for run_number in range(int(config['settings']['repeated_runs'])):
            train_loader = DataLoader(train_dataset, batch_size=int(config['model']['batch_size']), shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

            # Init model
            model = load_model(config['model']['model_architecture'],float(config['model']['dropout_percent']),num_classes)
            model = model.to(device)

            # Init optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),lr=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config['model']['epochs'])-1)
            scaler = torch.cuda.amp.GradScaler()


            CHECKPOINT = f"./{config['settings']['model_dir']}/{config['settings']['experiment_name']}_{config['model']['model_architecture']}_images_{num_images}_run_{run_number}.pth"

            print(f'Run {run_number}')
            print('--------------------------------')

            # Training loop
            train_acc_list = []
            val_acc_list = []
            train_loss_list = []
            val_loss_list = []
            val_loss_min = np.Inf

            for epoch in range(int(config['model']['epochs'])):
                model.train()
                train_loss = []
                train_running_corrects = 0
                val_running_corrects = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        train_loss.append(loss.item())

                        _, predicted = torch.max(outputs.data, 1)
                        #_,labels = torch.max(labels.data, 1)
                        train_running_corrects += torch.sum(predicted == labels.data)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()

                    #loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, epochs-1, optimizer.param_groups[0]['lr']))
                    #loop.set_postfix(loss=np.mean(train_loss))

                train_loss = np.mean(train_loss)
                train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset)

                model.eval()

                val_loss = 0

                # Validation loop
                with torch.cuda.amp.autocast(), torch.no_grad():    
                    for images, labels in valid_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        #_,labels = torch.max(labels.data, 1)
                        val_running_corrects += torch.sum(predicted == labels.data)

                val_loss /= len(valid_loader.dataset)
                val_epoch_acc = val_running_corrects.double() / len(valid_loader.dataset)

                print(f'Epoch {epoch}: train loss: {train_loss:.5f} | train acc: {train_epoch_acc:.3f} | val_loss: {val_loss:.5f} | val acc: {val_epoch_acc:.3f}')

                train_acc_list.append(train_epoch_acc.item())
                val_acc_list.append(val_epoch_acc.item())
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                if val_loss < val_loss_min:
                        print(f'Valid loss improved from {val_loss_min:.5f} to {val_loss:.5f} saving model to {CHECKPOINT}')
                        val_loss_min = val_loss
                        best_epoch = epoch
                        torch.save(model.state_dict(), CHECKPOINT)

                print(f'Best epoch {best_epoch} | val loss min: {val_loss_min:.5f}')

            run_train_acc[run_number] = train_acc_list
            run_val_acc[run_number] = val_acc_list
            run_train_loss[run_number] = train_loss_list
            run_val_loss[run_number] = val_loss_list

            # Delete model just to be sure
            del loss, model, optimizer
            torch.cuda.empty_cache()
        
        # Final Results Output
        avg = 0.0
        for fold,val in run_val_acc.items():
            print(f'Highest val_acc for fold {fold}: {np.max(val):.3f}')
            avg += np.max(val)
        print(f'Average for all folds: {avg/len(run_val_acc.items()):.3f}')
        
        #Saving Metrics
        df = pd.DataFrame()
        for metric,name in zip([run_train_acc,run_train_loss,run_val_acc,run_val_loss],['train_acc','train_loss','val_acc','val_loss']):
            dffold = pd.DataFrame.from_dict(metric,orient='columns')
            dffold.columns = [f'fold{x}_{name}' for x in range(len(metric))]
            dffold = dffold.rename_axis('epochs')
            df = pd.concat([df,dffold],axis=1)
        df.to_csv(f"./logs/{config['settings']['experiment_name']}_{config['model']['model_architecture']}_{num_images}_metrics.csv")
        #pd.DataFrame.from_dict([train_ids_dict,test_ids_dict],orient='columns').to_csv('./logs/minaug_cv_splits.csv')


if __name__ == "__main__":
    #args = parse_args()
    config = parse_config()
    main(config)