import dataset as d
from util.preprocessing import  *
import multiprocessing
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
import dataset as d
from train.models.encoder_resnet import EncoderResnet
from torch import optim
import pandas as pd
import argparse

from train.trainer import trainer



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


def main(config):

    seed_everything(config['seed']) # Seed 고정

    train_mean = (0.4194325, 0.3830166, 0.3490198)
    train_Std = (0.23905228, 0.2253936, 0.22334467)

    valid_mean = (0.4170096, 0.38036022, 0.34702352)
    valid_Std = (0.23896241, 0.22566794, 0.22329141)

    train_data = pd.read_csv('./data/open/train_data.csv')
    valid_data = pd.read_csv('./data/open/valid_data.csv')
    test_data = pd.read_csv('./data/open/test_data.csv')

    train_transform = d.ImageTransForm(config['img_size'], train_mean, train_Std)
    valid_transform = d.ImageTransForm(config['img_size'], valid_mean, valid_Std)

    train_dataset = d.CustomDataset(train_data, 'train', transform=train_transform)
    valid_dataset = d.CustomDataset(valid_data, 'valid', transform=valid_transform)



    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=config['num_worker'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_worker'], pin_memory=True)

    dataloader_dict = {'train': train_loader, 'valid': valid_loader}

    encoder = EncoderResnet(512)
    device = get_gpu(config['gpu_id'])
    print(device)
    torch.cuda.is_available()
    print(encoder)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    criterion.to(device)
    encoder.to(device)

    train_history, valid_history = trainer(encoder, dataloader_dict=dataloader_dict, criterion=criterion, num_epoch=config['epochs'], optimizer=optimizer, device=device, early_stop=config['early_stop'])
    return train_history, valid_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    args.seed = 41
    args.img_size = 224
    args.num_worker = multiprocessing.cpu_count()
    #
    #
    config = vars(args)
    #
    train_history, valid_history = main(config)

    # pd = pd.DataFrame(columns=['train_loss', 'test_loss'],
    #                   data=[(train, valid) for train, valid in zip(train_history, valid_history)])
    # pd.to_csv('loss.csv')
