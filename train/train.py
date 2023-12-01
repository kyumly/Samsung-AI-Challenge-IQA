import torch
from tqdm import tqdm

def train(model, train_dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    mask = 0
    model.train()
    for img, mos, comment in tqdm(train_dataloader):
        x = img.to(device)
        y = mos.to(device)

        optimizer.zero_grad()
        if type(model).__name__ == 'EncoderGoogleNet':
            mos1, mos2, mos3 = model(x, True)
            mos_loss1 = criterion(mos1.to(torch.float64), y.to(torch.float64))
            mos_loss2 = criterion(mos2.to(torch.float64), y.to(torch.float64))
            mos_loss3 = criterion(mos3.to(torch.float64), y.to(torch.float64))
            loss = mos_loss1 + 0.3 * (mos_loss2 * mos_loss3)
        else:
            out, mos_pred = model(x)
            loss = criterion(mos_pred.to(torch.float64), y.to(torch.float64))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


    return epoch_loss / len(train_dataloader)