from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, in_channels=3):
        super(EfficientNetB4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4') if pretrained else EfficientNet.from_name('efficientnet-b4')

        if in_channels != 3:
            out_channels = self.model._conv_stem.out_channels
            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

            # Initialize the new conv weights by averaging over RGB channels
            with torch.no_grad():
                new_conv.weight = nn.Parameter(self.model._conv_stem.weight.sum(dim=1, keepdim=True) / 3.0)

            self.model._conv_stem = new_conv

        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
