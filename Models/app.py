from flask import Flask
import torch

app = Flask("__name__")

model = torch.jit.load("Model_25_feb.pth")
model.eval()

@app.route("/encode")
def encode():
    pass