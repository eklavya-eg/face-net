import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.optim import lr_scheduler


class Model(nn.Module):
    def __init__(self, in_channels, embedding_size, p_dropout, p_linear_dropout):
        super(Model, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        

        self.b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.1))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, 92, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(92),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(92, 128, kernel_size=3, padding="same"),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.1),)
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, 24, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(24),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(24, 32, kernel_size=5, padding="same"),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.1))
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1, stride = 1),
                                nn.Conv2d(in_channels, 32, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.1))
        

        self.b5 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.1))
        self.b6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(128, 64, kernel_size=3, padding="same"),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.1),)
        self.b7 = nn.Sequential(nn.Conv2d(256, 24, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(24),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(24, 16, kernel_size=5, padding="same"),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.1))
        self.b8 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1, stride = 1),
                                nn.Conv2d(256, 16, kernel_size=1, padding="same"),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.1))

        self.dropout = nn.Dropout2d(p_dropout)
        
        self.fc = nn.Sequential(nn.Linear(128*56*56, 512, bias=True),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(p_linear_dropout),
                                nn.Linear(512, embedding_size, bias=True))
        
        
    def forward(self, x):
        x = self.conv(x)

        xb1 = self.b1(x)
        xb2 = self.b2(x)
        xb3 = self.b3(x)
        xb4 = self.b4(x)
        x = torch.cat([xb1, xb2, xb3, xb4], dim = 1)

        x = self.dropout(x)

        xb5 = self.b5(x)
        xb6 = self.b6(x)
        xb7 = self.b7(x)
        xb8 = self.b8(x)
        x = torch.cat([xb5, xb6, xb7, xb8], dim = 1)
        x = self.dropout(x)
        x = x.reshape(-1, 128*56*56)
        # x = x.contiguous.view(-1, 128*56*56)
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

class Schedular:
    def __init__(self, optimizer, lr, step_size, gamma):
        self.optimizer = optimizer
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
    
    def decay(self):
        return lr_scheduler.StepLR(self.optimizer, self.step_size, self.gamma)