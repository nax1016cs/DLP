import torch
import torch.nn as nn
import os
import json


def load_checkpoint(path, model, device):  
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict(),}
    torch.save(state_dict, path)

def Condition(filename):
    with open("dataset/objects.json", "r") as f:
        objects = json.load(f)

    with open("dataset/" + filename , "r") as f:
        labels = json.load(f)

    oneHotLabel = torch.zeros(len(labels),len(objects))
    for i in range(len(labels)):
        for condition in labels[i]:
            oneHotLabel[i, int(objects[condition])] = 1.
    return oneHotLabel

def sample(batch_size, latent_size):
    return torch.randn(batch_size, latent_size)