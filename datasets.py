import os
from typing import Callable, Tuple, Optional

import albumentations as albu
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from constants import *


class ImageDataset(Dataset):
    """Dataset for train classification model"""

    def __init__(self,
                 df: pd.DataFrame,
                 path: str,
                 transform: Optional[Callable] = None,
                 albu_trans: Optional[Callable] = None
                 ):
        super().__init__()
        self.path = path
        self.dataframe = df
        self.transform = transform
        self.albu_trans = albu_trans

    def __getitem__(self, x: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Returns label and Tensor of image """
        name, label = self.dataframe.iloc[x].values
        x = self.load_sample(name)
        if self.albu_trans:
            x = Image.fromarray(self.albu_trans(image=np.asarray(x))['image'])
        if self.transform:
            x = self.transform(x)
        return x, torch.LongTensor([label, ])

    def load_sample(self, name: str) -> Image:
        """Function for load image"""
        name = Image.open(os.path.join(self.path, name))
        name.load()
        return name

    def __len__(self) -> int:
        """Returns the length od dataset"""
        return len(self.dataframe)


class TestDataset(Dataset):
    """Dataset for test/validation classification model"""

    def __init__(self,
                 data_df: pd.DataFrame,
                 path: str,
                 transform: Optional[Callable] = None):
        self.data_df = data_df
        self.path = path
        self.transform = transform

    def __getitem__(self, idx: int):
        """Returns label and Tensor of image """
        image_name = self.data_df.iloc[idx].filename

        # читаем картинку
        image = self.load_sample(image_name)

        # преобразуем, если нужно
        if self.transform:
            image = self.transform(image)

        return image_name, image

    def load_sample(self, name: str) -> Image:
        """Function for load image"""
        name = Image.open(os.path.join(self.path, name))
        name.load()
        return name

    def __len__(self):
        """Returns the length od dataset"""
        return len(self.data_df)


def get_ablu_transform() -> Callable:
    alba = [
        albu.ImageCompression(quality_lower=60, quality_upper=100),
        albu.GaussNoise(p=0.5),
        albu.MotionBlur()

    ]
    return albu.Compose(alba)


def get_train_transform() -> Callable:
    """Function return train transforms for images"""
    tran = [

        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=(0.5, 3), contrast=0, saturation=0, hue=0),
        transforms.GaussianBlur(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ]
    return transforms.Compose(tran)


def get_exptrain_transform() -> Callable:
    """Function return experimental train transforms for images"""
    tran = [

        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=(0.5, 3), contrast=(0.4, 1.5), saturation=0, hue=0),
        transforms.GaussianBlur(1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ]
    return transforms.Compose(tran)


def get_test_transform() -> Callable:
    """Function return test transforms for images"""
    tran = [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(tran)


def get_label_replacers(df_path: str = TRAIN_DATAFRAME_PATH) -> Tuple[dict, dict]:
    """Return dictionaries for changing labels in dataframe"""
    label2int = {}
    int2label = {}
    data = pd.read_csv(df_path)
    was = sorted(data.label.unique())
    will = sorted(data.label.astype('category').cat.codes.unique())
    for i in range(len(will)):
        label2int[was[i]], int2label[will[i]] = will[i], was[i]

    return label2int, int2label


def get_train_loader(train_dataset: Dataset) -> DataLoader:
    """Returns train loader"""
    return DataLoader(dataset=train_dataset,
                      batch_size=32,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=2)


def get_test_loader(test_dataset: Dataset) -> DataLoader:
    """Returns test loader"""
    return DataLoader(dataset=test_dataset,
                      batch_size=32,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=2)
