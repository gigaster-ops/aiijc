from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from constants import *


def get_resnet_152(device: str = DEVICE,
                   ckpt_path: Optional[str] = None
                   ) -> nn.Module:
    """Returns the pretrained model resnet152 and if checkpoint is specified load it"""
    model = models.resnet152(True)
    model.fc = nn.Sequential(nn.Linear(2048, 131))
    model = model.to(device)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
        except:
            print("Wrong checkpoint")
    return model


def get_densenet_121(device: str = DEVICE,
                     ckpt_path: Optional[str] = None
                     ) -> nn.Module:
    """Returns the pretrained model densenet152 and if checkpoint is specified load it"""
    model = models.densenet121(True)
    model.classifier = nn.Sequential(nn.Linear(1024, 131))
    model = model.to(device)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
        except:
            print("Wrong checkpoint")
    return model


def get_vgg_19(device: str = DEVICE,
               ckpt_path: Optional[str] = None
               ) -> nn.Module:
    """Returns the pretrained model vgg19 and if checkpoint is specified load it"""
    model = models.vgg19(True)
    model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=4096, out_features=131, bias=True)

                                     )
    model = model.to(device)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
        except:
            print("Wrong checkpoint")
    return model
