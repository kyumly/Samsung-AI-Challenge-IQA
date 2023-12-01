import torch
from tqdm import tqdm


def train(model, train_dataloader, optimizer, criterion_dict, device, word2idx):
    epoch_loss = 0

    model.train()
    for img, mos, comment in tqdm(train_dataloader):
        img = img.to(device)
        mos = mos.to(device)
        comment = comment.to(device)

        comments_tensor = torch.zeros((len(comment), len(max(comment, key=len)))).long().to(device)
        for i, comment in enumerate(comment):
            tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
            comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])

        predicted_caption, predicted_mos = model(img, comment)
        caption_target = comments_tensor[:, 1:]

        loss_mos = criterion_dict['mos'](predicted_mos.to(torch.float64), mos.to(torch.float64))
        loss_caption = criterion_dict['caption'](predicted_caption.view(-1, cs.size(-1)), caption_target.reshape(-1))
        loss = loss_mos + loss_caption

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