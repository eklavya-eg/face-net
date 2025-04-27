import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        anchor, positive, negative = self.data[index]
        anchor = read_image(anchor, mode=ImageReadMode.RGB)
        positive = read_image(positive, mode=ImageReadMode.RGB)
        negative = read_image(negative, mode=ImageReadMode.RGB)
        return torch.as_tensor(anchor, dtype=torch.float32)/255.0, torch.as_tensor(positive, dtype=torch.float32)/255.0, torch.as_tensor(negative, dtype=torch.float32)/255.0
    
    def __len__(self):
        return len(self.data)