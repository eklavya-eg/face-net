import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import cv2
import dlib
import h5py
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Models.Variables import Variables
from Models.Model import Model

IS_DATABASE_CREATED = 0
DATABASE_PATH = "faces.h5"
MODEL_PATH = 'Models/Weights/model_epoch_7pth'
DEVICE = "cuda"
MARGIN = 0.3

class base:
    def fetch(encodings, fetcher):
        dist = []
        details = []
        keys = list(fetcher.keys())
        print(keys)
        if not keys: return "UNKNOWN"
        for i in keys:
            stored_encode = torch.tensor(fetcher.get(i), device=DEVICE)
            dist.append(torch.pairwise_distance(stored_encode, encodings))
            details.append(i)
        ind = torch.argmin(torch.tensor(dist))
        print(dist)
        print(details)
        return str(details[ind])

    def add(encodings, name, data_writer):
        with torch.no_grad():
            encodings = encodings.cpu().numpy()
        data_writer.create_dataset(name=name, data=encodings)
        

if __name__=="__main__":

    variables = Variables()
    model = Model(in_channels=variables.in_channels, embedding_size=variables.emembedding_size, p_dropout=variables.p_dropout, p_linear_dropout=variables.p_linear_dropout)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device=DEVICE)
    model.eval()
    if not IS_DATABASE_CREATED: data_writer = h5py.File(DATABASE_PATH, 'a')
    else: data_writer = h5py.File(DATABASE_PATH, 'a')
    fetcher = h5py.File(DATABASE_PATH, 'r')
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    while(True):
        user_in = input()
        if user_in=="add":
            name = input("NAME: ")
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            if (x<=0) or (y<=0) or (w<=0) or (h<=0): continue
            if (y-25<=0) or (y+h+25<=0) or (x-25<=0) or (x+w+25<=0): continue
            face = torch.permute(torch.unsqueeze(torch.tensor(cv2.resize(frame[y-25:y+h+25,x-25:x+w+25,:], (112, 112)), device=DEVICE, dtype=torch.float32), dim=0), (0,3,1,2))
            encoding = model.forward(face)
            encoding = torch.squeeze(encoding)
            base.add(encoding, name, data_writer)
            print("Face Added Succusfully...")

        elif user_in=="infer":
            while True:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                facesi = detector(gray)
                framecpy = frame.copy()
                for i in range(len(facesi)):
                    x, y, w, h = facesi[i].left(), facesi[i].top(), facesi[i].width(), facesi[i].height()
                    print(x, y, w, h)
                    if (x<=0) or (y<=0) or (w<=0) or (h<=0): continue
                    face = torch.permute(torch.unsqueeze(torch.tensor(cv2.resize(framecpy[y-25:y+h+25,x-25:x+w+25,:], (112, 112)), device=DEVICE, dtype=torch.float32), dim=0), (0,3,1,2))
                    encoding = model.forward(face)
                    name = base.fetch(encoding, fetcher)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    print(name)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == 27: break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()