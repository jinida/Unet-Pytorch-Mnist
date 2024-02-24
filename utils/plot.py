import os
from math import sqrt
import random
import numpy as np
import cv2
from utils.util import isSquareRootInteger

def generateColorByClass(numClass: int = 10) -> list[tuple[int, int, int]]:
    if numClass > 256 * 256 * 256:
        raise ValueError("Too many classes for unique RGB generation")

    colors = set()
    while len(colors) < numClass:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(color)

    return list(colors)

def getDrawingMaskOnImage(mask: np.ndarray, image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray: 
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness=1)

    colorMask = np.zeros_like(image)
    colorMask[mask == 255] = color
    result = cv2.addWeighted(image, 0.85, colorMask, 0.15, 0)
    return result

def getDrawingTitleImage(title: str, image: np.ndarray, textColor: tuple[int, int, int]) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.25
    fontThickness = 1
    titlePosition = (0, 6)
    image = cv2.putText(image, title, titlePosition, font, fontScale, textColor, fontThickness)
    return image

def saveImagesForMnistSegment(images, labelsByClass, numbers, colorByClass, threshold, savePath, fileName='segmentImage.jpg'):
    imageNum = len(images)
    if isSquareRootInteger(imageNum):
        imageRowNum = int(sqrt(imageNum))
    else:
        raise ValueError("Number of images is not appropriate.")
    
    imageSize = images[0].shape[1]
    saveImage = np.zeros((imageSize * imageRowNum, imageSize * imageRowNum, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        labelByClass = labelsByClass[idx]
        for classIdx, label in enumerate(labelByClass):
            label[label > threshold] = 255
            label[label <= threshold] = 0
            label = label.byte().cpu().numpy()
            image = getDrawingMaskOnImage(label, image, colorByClass[classIdx])
            
        i = int(idx % imageRowNum)
        j = int(idx / imageRowNum)
        saveImage[j * imageSize:j * imageSize + imageSize, i * imageSize:i * imageSize + imageSize] = image
    
    cv2.imwrite(os.path.join(savePath, fileName), saveImage)
    