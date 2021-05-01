import torch.nn as nn
import math

"""
Imported by https://github.com/lukemelas/EfficientNet-PyTorch
"""

from efficientnet_pytorch import EfficientNet

class EfficientNet(nn.Module):
    def __init__(self):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')


    def forward(self, x):
        print("Hey")
        return self.model.extract_features

