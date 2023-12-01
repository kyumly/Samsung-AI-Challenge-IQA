from train.models.encoder_resnet import EncoderResnet
from train.models.encoder_googlenet import EncoderGoogleNet
import os
import torch
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_gpu(gpu_id):
    if gpu_id  == 1 and torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif gpu_id == 2 and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_encoder(model_name, output):
    if model_name == "resnet":
        return EncoderResnet(output)
    elif model_name == "google":
        return EncoderGoogleNet(output)
    else:
        return EncoderResnet(output)
