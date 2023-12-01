import torch
def evaluate(model, test_dataloader, criterion_dict, device):
    epoch_loss = 0
    epoch_acc = 0
    mask = 0

    model.eval()
    with torch.no_grad():
        for img, mos, comment in test_dataloader:
            img = img.to(device)
            mos = mos.to(device)
            comment = comment.to(device)

            comments_tensor = torch.zeros((len(comments), len(max(comments, key=len)))).long().to(device)
            for i, comment in enumerate(comments):
                tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
                comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])

            predicted_caption, predicted_mos = model(imgs, comments)
            caption_target = comments_tensor[:,1:]

            loss_mos = criterion_dict['mos'](predicted_mos.to(torch.float64), mos.to(torch.float64))
            loss_caption = criterion_dict['caption'](predicted_caption.view(-1, cs.size(-1)), caption_target.reshape(-1))
            loss = loss_mos + loss_caption
            epoch_loss += loss.item()

            mask += 1
            if mask == 100:
                break

    return epoch_loss / len(test_dataloader)