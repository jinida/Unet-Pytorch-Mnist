import torch
from torch import nn
from model.common import Conv, ConvTranspose, Concat

''' For MNIST'''
class ContractPath(nn.Module):
    def __init__(self, inChannelNum, outChannelNum):
        super().__init__()
        self.conv1 = Conv(inChannelNum, outChannelNum, 3)
        self.conv2 = Conv(outChannelNum, outChannelNum, 3)
        self.maxPool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.maxPool(x), x

class ExtractPath(nn.Module):
    def __init__(self, inChannelNum, outChannelNum):
        super().__init__()
        self.upConv = ConvTranspose(inChannelNum, outChannelNum)
        self.cat = Concat()
        self.conv1 = Conv(inChannelNum, outChannelNum, 3)
        self.conv2 = Conv(outChannelNum, outChannelNum, 3)
        
    def forward(self, previousX, x):
        return self.conv2(self.conv1(self.cat([previousX, self.upConv(x)])))
    
class UNet(nn.Module):
    def __init__(self, pathBlockNum, classNum=10):
        super().__init__()
        self.contractPaths = nn.ModuleList()
        self.extractPaths = nn.ModuleList()
        
        preChannelNum = 64
        self.contractPaths.append(ContractPath(1, preChannelNum))
        for _ in range(pathBlockNum - 1):
            self.contractPaths.append(ContractPath(preChannelNum, preChannelNum * 2))
            preChannelNum *= 2
        
        self.extraConv1 = Conv(preChannelNum, preChannelNum * 2)
        preChannelNum *= 2
        self.extraConv2 = Conv(preChannelNum, preChannelNum)

        for _ in range(pathBlockNum):
            self.extractPaths.append(ExtractPath(preChannelNum, preChannelNum // 2))
            preChannelNum //= 2

        self.fullyConv = Conv(preChannelNum, classNum, 3, hasActivation=False)
    
    def forward(self, x):
        featureMaps = list()
        for path in self.contractPaths:
            x, saveFeature = path(x)
            featureMaps.append(saveFeature) 
        x = self.extraConv2(self.extraConv1(x))
        for path, previousX in zip(self.extractPaths, reversed(featureMaps)):
            x = path(previousX, x)
        
        return self.fullyConv(x)
        
        
        
            