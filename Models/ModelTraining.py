import pandas as pd
import numpy as np
import os
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
    warnings.filterwarnings("ignore")

    path = "Models/Data/Dataset.csv"
    data = pd.read_csv(path)
    data = data.values
    data = data[:2000]
    # np.random.shuffle(data)
    train_len = int(data.__len__()*80/100)
    train = data[0:train_len]

    val_start = int((len(data)-train_len)/2)
    val = data[train_len:val_start+train_len]
    test = data[val_start+train_len:len(data)]

    train = FaceDataset(train)
    val = FaceDataset(val)
    test = FaceDataset(test)
    variables = Variables()
    print("Samples")
    print("\ttrain:", len(train))
    print("\tval:", len(val))
    print("\ttest:", len(test))
    train = DataLoader(train, batch_size = 4, shuffle=False)
    val = DataLoader(val, batch_size = 4, shuffle=False)
    test = DataLoader(test, batch_size = 4, shuffle=False)
    test_batch_change = DataLoader(test, batch_size = 8, shuffle=False)
    print("Batches")
    print("\ttrain:", len(train))
    print("\tval:", len(val))
    print("\ttest:", len(test))
    
    
    
    model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, p_dropout=variables.p_dropout, p_linear_dropout=variables.p_linear_dropout)
    tripletloss = TripletLoss(margin=variables.margin, p=variables.p)
    optimizer = Optimizer(model=model, lr=variables.lr).optimize()
    schedular = Schedular(optimizer, variables.lr, variables.step_size, variables.gamma).decay()
    device = "cuda"
    model = model.to(device)

    # Load Training Cache if available
    cache = os.listdir("Models/Training Cache")
    cache_load = False
    if "model_cache.pth" not in cache:
        epoch_start = 0
    else:
        cache_load = True
        model.train()
        training_ckpt = torch.load(os.path.join("Models/Training Cache", "model_cache.pth"))
        train_iter = enumerate(train)
        cache_train_loop = tqdm(train_iter, leave=False)
        for i, j in cache_train_loop:
            if i==training_ckpt['batch']: break
        epoch_start = training_ckpt['epoch']
        model.load_state_dict(training_ckpt['model_state_dict'])
        optimizer.load_state_dict(training_ckpt['optimizer_state_dict'])
        schedular.load_state_dict(training_ckpt['scheduler_state_dict'])
        train_batch_cost = training_ckpt['train_loss']

    try:
        for epoch in range(epoch_start, variables.epochs):
                model.train()
                if cache_load:
                    cache_load=False
                    loop = tqdm(train_iter, total=len(train)-training_ckpt['batch']-1, leave=True)
                    train_cost = train_batch_cost
                else:
                    train_cost = 0.0
                    loop = tqdm(enumerate(train), total=len(train), leave=True)
                for batch_number, train_batch in loop:
                    anc, pos, neg = train_batch
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                    optimizer.zero_grad()
                    anc = model(anc)
                    pos = model(pos)
                    neg = model(neg)
                    loss = tripletloss.loss(anc, pos, neg)
                    train_cost += loss.item()
                    loss.backward()
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"{name} --> Grad shape: {param.grad.shape}, Grad mean: {param.grad.mean().item()}")
                    #     else:
                    #         print(f"{name} --> No gradient yet")
                    # exit()
                    optimizer.step()
                    loop.set_description(f"[{epoch+1}/{variables.epochs}]")
                    loop.set_postfix(loss=loss.item())

                    # Caching for Training Pause and Resume
                    checkpoint = {
                        'epoch': epoch,
                        'batch': batch_number,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': schedular.state_dict(),
                        'train_loss': train_cost,
                    }
                avg_train_loss = train_cost/len(train)
                

                model.eval()
                with torch.no_grad():
                    val_cost = 0.0
                    loop_val = tqdm(enumerate(val), total=len(val), leave=True)
                    for _, val_batch in loop_val:
                        anc, pos, neg = val_batch
                        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                        anc = model(anc)
                        pos = model(pos)
                        neg = model(neg)
                        val_loss = tripletloss.loss(anc, pos, neg)
                        val_cost += val_loss.item()
                        loop_val.set_description(f"[{epoch+1}/{variables.epochs}]")
                        loop_val.set_postfix(loss=val_loss.item())
                    avg_val_loss = val_cost/len(val)
                    print(f"\nEpoch {epoch+1}/{variables.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
                checkpoint_filename = f"model_epoch_{epoch+1}.pth"
                if avg_train_loss < 0.2:
                    if avg_val_loss < 0.2:
                        torch.save(model.state_dict(), checkpoint_filename)
                elif (epoch+1)%10==0:
                    torch.save(model.state_dict(), checkpoint_filename)
                schedular.step()
                print("\n")
    except KeyboardInterrupt:
        torch.save(checkpoint, f'Models/Training Cache/model_cache.pth')