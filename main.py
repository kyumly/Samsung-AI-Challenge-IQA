import dataset as d
from util.preprocessing import  *
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import dataset as d

from torch import optim
import pandas as pd
import argparse

from train.trainer import trainer
from util.common import seed_everything, get_encoder, get_gpu


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

    encoder = get_encoder(config['model_name'], 512)
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
    parser.add_argument('--model_name', type=str, default='resnet')

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
