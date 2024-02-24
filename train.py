import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

import math

from utils.util import createIncrementalPath
from datasets.MNISTMaskDataset import MNISTMaskDataset
from model.unet import UNet
from utils.plot import saveImagesForMnistSegment, generateColorByClass

torch.autograd.set_detect_anomaly(True)

batchSize = 64
imgSize = 28
labelNum = 10
threshold = .8
plot = True
colorByClass = generateColorByClass(labelNum)

learningRate = 1e-3
numEpoch = 100

resultPath = createIncrementalPath('./result')
os.makedirs("./data/MNIST", exist_ok=True)
transform = transforms.Compose([transforms.Resize(imgSize), transforms.ToTensor()])
trainDataset = MNISTMaskDataset("./data/MNIST", train=True, download=True, transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
trainLoaderSize = len(trainLoader)

testDataset = MNISTMaskDataset("./data/MNIST", train=False, download=True, transform=transform)
testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, drop_last=True)
testLoaderSize = len(testLoader)

unet = UNet(2, labelNum).cuda()
optimizer = torch.optim.Adam(unet.parameters(), lr=learningRate, amsgrad=True)
bceLossFunc = nn.BCEWithLogitsLoss()

print("Train Start!")
bestLoss = math.inf
for epoch in range(numEpoch):
    unet.train()
    totalLoss = 0.
    for i, (images, labels, _) in enumerate(trainLoader):
        images = images.cuda()
        labels = labels.cuda()
        predict = unet(images)
        
        loss = bceLossFunc(predict, labels)
        totalLoss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    trainLoss = totalLoss / trainLoaderSize
    
    with torch.no_grad():
        unet.eval()
        totalLoss = 0.
        if plot:
            images, labels, numbers = next(iter(testLoader))
            predict = unet(images.cuda()).sigmoid()
            saveImagesForMnistSegment(images, predict, numbers, colorByClass, threshold, resultPath, f'epoch{epoch}.jpg')
            for images, labels, numbers in testLoader:
                images = images.cuda()
                labels = labels.cuda()
                predict = unet(images)
                
                loss = bceLossFunc(predict, labels)
                totalLoss += loss.item()
            
        else:
            for images, labels, numbers in testLoader:
                images = images.cuda()
                labels = labels.cuda()
                predict = unet(images)
                
                loss = bceLossFunc(predict, labels)
                totalLoss += loss.item()
    
    if totalLoss < bestLoss:
        torch.save(unet.state_dict(), os.path.join(resultPath, 'best.pt'))
        bestLoss = totalLoss
    print(f"epoch: {epoch + 1}, Train Loss: {trainLoss}, Test Loss: {totalLoss / testLoaderSize}")