import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Variables import Variables
from DataLoader import FaceDataset
from Model import Model, Optimizer, TripletLoss, Schedular

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.init")

    variables = Variables()
    model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, p_dropout=variables.p_dropout, p_linear_dropout=variables.p_linear_dropout)
    model.load_state_dict(torch.load('Models/Weights/model_epoch_3.pth'))

    model_script = torch.jit.script(model)
    model_script.save("Model_25_feb.pth")