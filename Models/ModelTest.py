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

def accurate(anc, pos, neg, margin):
    anc_pos = torch.pairwise_distance(anc, pos, p=2).mean()
    anc_neg = torch.pairwise_distance(anc, neg, p=2).mean()
    if anc_neg-anc_pos >= margin:
        return True
    else:
        return False

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.init")

    path = "Models/Data/Dataset.csv"
    data = pd.read_csv(path)
    data = data.values
    np.random.shuffle(data)
    test = data[:40000]
    

    test = FaceDataset(test)
    variables = Variables()
    test = DataLoader(test, batch_size = variables.batch_size_test_val, shuffle=False)

    print("test:", len(test))
    
    
    
    device = "cuda"
    model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, p_dropout=variables.p_dropout, p_linear_dropout=variables.p_linear_dropout)
    tripletloss = TripletLoss(margin=variables.margin, p=variables.p)
    model.load_state_dict(torch.load('Models/Weights/model_epoch_3.pth'))
    model.to(device)
    model.eval()
    test_margin = 0.2

    with torch.no_grad():
        cost = 0.0
        accuracy = 0
        loop_val = tqdm(enumerate(test, 1), total=len(test), leave=True)
        for n, val_batch in loop_val:
            anc, pos, neg = val_batch
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            anc = model.forward(anc, variables.batch_size_test_val)
            pos = model.forward(pos, variables.batch_size_test_val)
            neg = model.forward(neg, variables.batch_size_test_val)
            accuracy += accurate(anc, pos, neg, test_margin)
            loss = tripletloss.loss(anc, pos, neg)
            cost += loss.item()
            loop_val.set_postfix(loss=loss.item(), accuracy=accuracy/n)
        avg_loss = cost/len(test)
print(f"average loss: {str(avg_loss)}")
print("accuracy:", str((accuracy*100)/n)+"%")