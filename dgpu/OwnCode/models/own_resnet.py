import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet152', 'resnet101', 'resnet50', 'resnet34', 'resnet18']

model_links = {
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3(p_in, p_out):
    # 3x3 convolution with padding
    return nn.Conv2d(p_in, p_out, kernel_size=3, stride=1, padding=1, bias=False)

def conv1(p_in, p_out):
    # 1x1 convolution
    return nn.Conv2d(p_in, p_out, kernel_size=1, stride=1, bias=False)

class BasicBlock(nn.Module):
    exp = 1

