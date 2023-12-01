import torch
from tqdm import tqdm


def evaluate(model, valid_dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for img, mos, comment in tqdm(valid_dataloader):
            x = img.to(device)
            y = mos.to(device)
            if type(model).__name__ == 'EncoderGoogleNet':
                mos_pred  = model(x)
            else:
                out, mos_pred = model(x)

            loss = criterion(mos_pred.to(torch.float64), y.to(torch.float64))
            epoch_loss += loss.item()

    return epoch_loss / len(valid_dataloader)

def test(model, valid_dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    loss_history = []
    with torch.no_grad():
        for img, mos, comment in tqdm(valid_dataloader):
            x = img.to(device)
            y = mos.to(device)
            if type(model).__name__ == 'EncoderGoogleNet':
                mos_pred  = model(x)
            else:
                out, mos_pred = model(x)

            loss = criterion(mos_pred.to(torch.float64), y.to(torch.float64))
            loss_history.append(loss)
            epoch_loss += loss.item()

    return epoch_loss / len(valid_dataloader), loss_history