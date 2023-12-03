import torch
import torch.nn as nn
import torchvision.models as models



def get_model(models):
    return list(models.children())[:-1]
#
class EncoderVGG(nn.Module):
    def __init__(self, output) -> None:
        super().__init__()
        model_block =get_model(models.vgg19_bn())
        self.cnn = nn.Sequential(*model_block)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(1024, output)
        self.mos = nn.Linear(1024, 1)



    def forward(self, img):
        features = self.cnn(img)
        print(features.shape)
        features = features.view(features.size(0), -1)
        print(features.shape)
        classifier = self.classifier(features)

        mos = self.mos(classifier)
        out = self.out(classifier)

        return out, mos
