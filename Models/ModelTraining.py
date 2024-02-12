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

    path = "Models/Data/Dataset.csv"
    data = pd.read_csv(path)
    data = data.values
    np.random.shuffle(data)
    train = data[0:70000]
    val = data[70000:72000]
    test = data[72000:75000]
    

    train = FaceDataset(train)
    val = FaceDataset(val)
    test = FaceDataset(test)
    variables = Variables()
    train = DataLoader(train, batch_size = variables.batch_size, shuffle=False)
    val = DataLoader(val, batch_size = variables.batch_size_test_val, shuffle=False)
    test = DataLoader(test, batch_size = variables.batch_size_test_val, shuffle=False)

    print("train:", len(train))
    print("val:", len(val))
    print("test:", len(test))
    
    
    
    model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, p_dropout=variables.p_dropout, p_linear_dropout=variables.p_linear_dropout)
    tripletloss = TripletLoss(margin=variables.margin, p=variables.p)
    optimizer = Optimizer(model=model, lr=variables.lr).optimize()
    schedular = Schedular(optimizer, variables.lr, variables.step_size, variables.gamma).decay()
    device = "cuda"
    # device = xm.xla_device()
    model.to(device)

    for epoch in range(0, variables.epochs):
        model.train()
        train_cost = 0.0
        loop = tqdm(enumerate(train), total=len(train), leave=True)
        for _, train_batch in loop:
            anc, pos, neg = train_batch
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            anc = model.forward(anc, variables.batch_size)
            pos = model.forward(pos, variables.batch_size)
            neg = model.forward(neg, variables.batch_size)
            loss = tripletloss.loss(anc, pos, neg)
            train_cost += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"[{epoch+1}/{variables.epochs}]")
            loop.set_postfix(loss=loss.item())
        avg_train_loss = train_cost/len(train)
        
        model.eval()
        with torch.no_grad():
            val_cost = 0.0
            loop_val = tqdm(enumerate(val), total=len(val), leave=True)
            for _, val_batch in loop_val:
                anc, pos, neg = val_batch
                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                anc = model.forward(anc, variables.batch_size_test_val)
                pos = model.forward(pos, variables.batch_size_test_val)
                neg = model.forward(neg, variables.batch_size_test_val)
                val_loss = tripletloss.loss(anc, pos, neg)
                val_cost += val_loss.item()
                loop_val.set_description(f"[{epoch+1}/{variables.epochs}]")
                loop_val.set_postfix(loss=val_loss.item())
            avg_val_loss = val_cost/len(val)
            print(f"\nEpoch {epoch+1}/{variables.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        checkpoint_filename = f"model_epoch_{epoch+1}.pth"
        if avg_train_loss < 1:
            if avg_val_loss < 1:
                torch.save(model.state_dict(), checkpoint_filename)
        elif (epoch+1)%5==0:
            torch.save(model.state_dict(), checkpoint_filename)
        schedular.step()
        print("\n")