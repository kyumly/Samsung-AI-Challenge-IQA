import torch
from tqdm import tqdm

from train.models.seq2seq import Seq2seq


def evaluate(model, valid_dataloader, criterion_dict, device, word2idx):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for img, mos, comment in tqdm(valid_dataloader):
            img = img.to(device)
            mos = mos.to(device)

            comments_tensor = torch.zeros((len(comment), len(max(comment, key=len)))).long()
            for i, comment in enumerate(comment):
                tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
                comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])

            comments_tensor = comments_tensor.to(device)

            if type(model.encoder).__name__ == 'EncoderGoogleNet':
                predicted_mos = model.encoder(img)
                loss = criterion_dict['mos'](predicted_mos.to(torch.float64), mos.to(torch.float64))
            else:
                predicted_caption, predicted_mos = model(img, comments_tensor)
                caption_target = comments_tensor[:,1:]
                loss_mos = criterion_dict['mos'](predicted_mos.to(torch.float64), mos.to(torch.float64))
                loss_caption = criterion_dict['caption'](predicted_caption.view(-1, 41100), caption_target.reshape(-1))
                loss = loss_mos + loss_caption

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