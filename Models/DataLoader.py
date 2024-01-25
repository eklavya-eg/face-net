import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data, batch, batch_size):
        position = batch_size*(batch-1)
        data = data.iloc[position:position+batch_size, :]
        dataImg = []
        for i in range(len(data)):
            images = []
            for j in data:
                images.append((read_image(data.iloc[i, j], ImageReadMode.RGB))/255.0)
            dataImg.append(images)
        return torch.asarray(dataImg, torch.float32).to('cuda')