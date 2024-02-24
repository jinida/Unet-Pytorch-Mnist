from torchvision.datasets import MNIST
from typing import Any
from PIL import Image
import torch

class MNISTMaskDataset(MNIST):
    def __init__(self, rootPath, train=True, download=True, transform=None) -> None:
        super().__init__(rootPath, train, transform, download)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, targetIdx = self.data[index], int(self.targets[index])
        
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
        
        maskImg = torch.repeat_interleave(torch.zeros_like(img, dtype=torch.float32), 10, 0)
        maskImg[targetIdx] = img != 0
        
        return img, maskImg, targetIdx