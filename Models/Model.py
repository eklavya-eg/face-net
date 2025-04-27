import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.optim import lr_scheduler


class Model(nn.Module):
    def __init__(self, in_channels, embedding_size, p_dropout, p_linear_dropout):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 192, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, padding=0),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.Conv2d(192, 384, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(384, 384, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(384, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
        )
        
        self.fc = nn.Sequential(nn.Linear(256*7*7, 640, bias=True),
                                nn.BatchNorm1d(640),
                                nn.LeakyReLU(0.1),
                                nn.Linear(640, embedding_size, bias=True))
        
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 256*7*7)
        x = self.fc(x)
        return x

class TripletLoss:
    def __init__(self, margin, p):
        self.margin = margin
        self.p = p

    def loss(self, anchor, positive, negative):
        distance_ap = F.pairwise_distance(anchor, positive, p = self.p)
        distance_an = F.pairwise_distance(anchor, negative, p = self.p)
        tripletloss = torch.clamp(input=distance_ap+(self.margin)-distance_an, min=0.0).mean()
        return tripletloss

class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
    
    def optimize(self):
        return optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-5)

class Schedular:
    def __init__(self, optimizer, lr, step_size, gamma):
        self.optimizer = optimizer
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
    
    def decay(self):
        return lr_scheduler.StepLR(self.optimizer, 20, 0.1)