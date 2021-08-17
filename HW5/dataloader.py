from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch

class CLEVRDataset(Dataset):
    def __init__(self,img_path,json_path):
        self.labels = torch.load('labels.pth')
        self.images = torch.load('images.pth')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

