import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, in_channels, embedding_size):
        super(Model, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        

        self.b1 = nn.Conv2d(3, 64, kernel_size=1, padding="same")
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, 92, kernel_size=1, padding="same"),
                                nn.Conv2d(92, 128, kernel_size=3, padding="same"))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, 24, kernel_size=1, padding="same"),
                                nn.Conv2d(24, 32, kernel_size=5, padding="same"))
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding="same", stride = 1),
                                nn.Conv2d(in_channels, 32, kernel_size=1, padding="same"))
        

        self.b5 = nn.Conv2d(256, 32, kernel_size=1, padding="same")
        self.b6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding="same"),
                                nn.Conv2d(128, 64, kernel_size=3, padding="same"))
        self.b7 = nn.Sequential(nn.Conv2d(256, 24, kernel_size=1, padding="same"),
                                nn.Conv2d(24, 16, kernel_size=5, padding="same"))
        self.b8 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding="same", stride = 1),
                                nn.Conv2d(256, 16, kernel_size=1, padding="same"))

        self.b5 = nn.Conv2d(256, 32, kernel_size=1, padding="same")
        self.b6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding="same"),
                                nn.Conv2d(128, 64, kernel_size=3, padding="same"))
        self.b7 = nn.Sequential(nn.Conv2d(256, 24, kernel_size=1, padding="same"),
                                nn.Conv2d(24, 16, kernel_size=5, padding="same"))
        self.b8 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding="same", stride = 1),
                                nn.Conv2d(256, 16, kernel_size=1, padding="same"))
        
        self.fc = nn.Sequential(nn.Linear(401408, 512, bias=True),
                                nn.Linear(512, embedding_size, bias=True))
        
    def forward(self, x):
        x = self.conv(x)

        xb1 = self.b1(x)
        xb2 = self.b2(x)
        xb3 = self.b3(x)
        xb4 = self.b4(x)
        x = torch.cat([xb1, xb2, xb3, xb4], dim = 1)

        xb5 = self.b5(x)
        xb6 = self.b6(x)
        xb7 = self.b7(x)
        xb8 = self.b8(x)
        x = torch.cat([xb5, xb6, xb7, xb8], dim = 1)

        x = x.view()
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
        return optim.Adam(self.model.parameters(), lr=self.lr)
