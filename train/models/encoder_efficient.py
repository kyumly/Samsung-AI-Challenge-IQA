import torch.nn as nn
import torchvision.models as models

from util.common import get_backbone


class Encoderefficient(nn.Module):
    def __init__(self, output):
        super().__init__()
        modules = get_backbone('efficient')
        self.cnn = nn.Sequential(*modules)
        #
        self.fc1 = nn.Linear(1792, 1024)
        # #
        self.out = nn.Linear(1024, output)
        #
        self.mos = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, img):
        features = self.cnn(img)
        features = features.view(features.size(0), -1)
        feature_out = self.relu(self.fc1(features))

        mos = self.mos(feature_out)
        out = self.out(feature_out)

        return out, mos

# model = Encoderefficient(512)
# #a = models.efficientnet_b4()
# #b = models.resnet50()
# ##k = list(a.children())[:-1]
# #c = nn.Sequential(*k)
# print(model)
# from torchsummary import summary
# print(summary(model, input_size=(3, 224,224), batch_size=1))




