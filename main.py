import dataset as d
from util.preprocessing import  *
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import dataset as d

from train.models.encoder_resnet import EncoderResnet
from train.models.decoder_seq import DecoderSeq
from train.models.seq2seq import Seq2seq
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

    all_data = pd.read_csv('./data/open/train.csv')
    train_data = pd.read_csv('./data/open/train_data.csv')
    valid_data = pd.read_csv('./data/open/valid_data.csv')
    test_data = pd.read_csv('./data/open/test_data.csv')

    train_transform = d.ImageTransForm(config['img_size'], train_mean, train_Std)
    valid_transform = d.ImageTransForm(config['img_size'], valid_mean, valid_Std)

    train_dataset = d.CustomDataset(train_data, 'train', transform=train_transform)
    valid_dataset = d.CustomDataset(valid_data, 'valid', transform=valid_transform)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=config['num_worker'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_worker'], pin_memory=True)
    all_comments = ' '.join(all_data['comments']).split()
    vocab = set(all_comments)
    vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    hidden_dim = 512
    embed_dim = 256
    output_dim = len(vocab)
    num_layers = 1

    device = get_gpu(config['gpu_id'])
    print(device)

    encoder = get_encoder(config['model_name'], hidden_dim)
    decoder = DecoderSeq(output_dim, embed_dim, hidden_dim, num_layers)
    model = Seq2seq(encoder, decoder, device)
    print(model)

    criterion_mos = nn.MSELoss()
    criterion_caption = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    criterion_mos.to(device)
    criterion_caption.to(device)
    model.to(device)

    dataloader_dict = {'train': train_loader, 'valid': valid_loader}
    criterion_dict = {'mos': criterion_mos, 'caption': criterion_caption}

    train_history, valid_history = trainer(model, dataloader_dict=dataloader_dict, criterion_dict=criterion_dict,
                                           num_epoch=config['epochs'], optimizer=optimizer, device=device,
                                           early_stop=config['early_stop'], word2idx=word2idx)
    return train_history, valid_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model_name', type=str, default='resnet')

    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()
    args.seed = 41
    args.img_size = 224
    args.num_worker = multiprocessing.cpu_count()
    #
    #
    config = vars(args)
    #
    train_history, valid_history = main(config)

    pd = pd.DataFrame(columns=['train_loss', 'test_loss'],
                      data=[(train, valid) for train, valid in zip(train_history, valid_history)])
    pd.to_csv('loss.csv', encoding='cp949', index=False)