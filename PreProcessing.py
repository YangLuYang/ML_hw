import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import keras as ks
import cv2


def readData():
    # data = pd.read_csv('./data/train(3).csv')
    data = pd.read_csv('./data/hw3.csv')
    labels = np.array(data['label'])
    features = np.array(data['feature'])
    images = []
    for i in range(features.shape[0]):
        images.append(features[i].split(' '))
    images = np.array(images, dtype=float).reshape(features.shape[0], 48, 48, 1)
    labels = ks.utils.to_categorical(labels, 7)
    cv2.imwrite('0.jpg', images[0])
    return images, labels

x, y = readData()