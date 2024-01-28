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
from Model import Model, Optimizer, TripletLoss

# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.init")

path = "Models/Data/Dataset.csv"
data = pd.read_csv(path)
data = data.values
np.random.shuffle(data)
train = data[0:7000]
val = data[7000:72000]
test = data[72000:77000]


train = FaceDataset(train)
val = FaceDataset(val)
test = FaceDataset(test)
variables = Variables()
train = DataLoader(train, batch_size = variables.batch_size, shuffle=False)
val = DataLoader(val, batch_size = variables.batch_size, shuffle=False)
test = DataLoader(test, batch_size = variables.batch_size, shuffle=False)


model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, batch_size=variables.batch_size)
tripletloss = TripletLoss(margin=variables.margin, p=variables.p)
optimizer = Optimizer(model=model, lr=variables.lr).optimize()
model.to("cuda")

for epoch in range(variables.epochs):
    model.train()
    train_cost = 0.0
    for train_batch in tqdm(train):
        anc, pos, neg = train_batch
        anc = model.forward(anc)
        pos = model.forward(pos)
        neg = model.forward(neg)
        loss = tripletloss.loss(anc, pos, neg)
        train_cost += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_cost/len(train)
    
    model.eval()
    with torch.no_grad():
        val_cost = 0.0
        for val_batch in val:
            anc, pos, neg = val_batch
            anc = model.forward(anc)
            pos = model.forward(pos)
            neg = model.forward(neg)
            val_loss = tripletloss.loss(anc, pos, neg)
            val_cost += val_loss.item()
        avg_val_loss = val_cost/len(val)
        print(f"Epoch {epoch + 1}/{variables.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")