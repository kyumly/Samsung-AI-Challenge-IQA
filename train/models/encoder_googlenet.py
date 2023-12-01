import torch
import torch.nn as nn
import torchvision.models as models


class EncoderGoogleNet(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.cnn_backbone = models.googlenet(aux_logits=True, num_classes= output)
        self.mos1 = nn.Linear(output, 1)
        self.mos2 = nn.Linear(output, 1)
        self.mos3 = nn.Linear(output, 1)

    def forward(self, img):
        out1, out2, out3 = self.cnn_backbone(img)

        mos_out1 = self.mos1(out1)
        mos_out2 = self.mos2(out2)
        mos_out3 = self.mos3(out3)

        return mos_out1, mos_out2, mos_out3
