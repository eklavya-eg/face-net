import torch
from torch import nn as nn
from torch import optim as optim


class Model(nn.module):
    batch_size = 48
    def __init__(self, in_channels):
        super(Model, self).__inti__()
        self.