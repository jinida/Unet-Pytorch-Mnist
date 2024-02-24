from torch import nn
import torch

def getAutoPaddingSize(kernelSize, paddingSize=None):
    if paddingSize is None:
        paddingSize = kernelSize // 2 if isinstance(kernelSize, int) else [x // 2 for x in kernelSize]
    return paddingSize

class Conv(nn.Module):
    def __init__(self, inChannelNum, outChannelNum, kernelSize=1, strideSize=1, paddingSize=None, groupSize=1, hasActivation=True):
        super().__init__()
        self.conv = nn.Conv2d(inChannelNum, outChannelNum, kernelSize, strideSize, getAutoPaddingSize(kernelSize, paddingSize), groups=groupSize, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannelNum)
        self.activation = nn.ReLU() if hasActivation is True else (hasActivation if isinstance(hasActivation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.batchNorm(self.conv(x)))

    def forward_fuse(self, x):
        return self.activation(self.conv(x))
    
class ConvTranspose(nn.Module):
    def __init__(self, inChannelNum, outChannelNum, kernelSize=2, strideSize=2, paddingSize=0, hasBatchNorm=True, hasActivation=True):
        super().__init__()
        self.convTranspose = nn.ConvTranspose2d(inChannelNum, outChannelNum, kernelSize, strideSize, paddingSize, bias=not hasBatchNorm)
        self.batchNorm = nn.BatchNorm2d(outChannelNum) if hasBatchNorm else nn.Identity()
        self.activation = nn.ReLU() if hasActivation is True else (hasActivation if isinstance(hasActivation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.batchNorm(self.convTranspose(x)))

    def forward_fuse(self, x):
        return self.activation(self.convTranspose(x))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
