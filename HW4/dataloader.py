import json
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_token = 1

class Dictionary:

    def __init__(self):
        self.char2idx = {'SOS': 0 , 'EOS': 1}
        self.idx2char = {0: 'SOS' , 1: 'EOS' }
        self.tense2idx={'sp':0 , 'tp':1, 'pg':2, 'p':3}
        self.idx2tense={0:'sp' , 1: 'tp', 2:'pg', 3:'p'}
        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        for c in alphabets:
            if c in self.char2idx:
                continue
            idx = len(self.char2idx)
            self.char2idx[c] = idx
            self.idx2char[idx] = c

    def encode(self, w):
        return torch.tensor(
              [ self.char2idx[c] for c in w ]
            + [ EOS_token ],
        device=device).view(-1, 1)

    def decode(self, t):
        word = []
        for char in t.view(-1):
            word.append(self.idx2char[char.item()])
        return ''.join(word)


class TrainDataset(Dataset):
    
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=np.str).reshape(-1)
        self.dict = Dictionary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index % len(self.dict.tense2idx)
    
class TestDataset(Dataset):

    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=np.str)
        self.dict = Dictionary()
        self.target = [
            ['sp',  'p'],
            ['sp',  'pg'],
            ['sp',  'tp'],
            ['sp',  'tp'],
            ['p',   'tp'],
            ['sp',  'pg'],
            ['p',   'sp'],
            ['pg', 'sp'],
            ['pg', 'p'],
            ['pg', 'tp']
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        return self.data[index][0], self.dict.tense2idx[self.target[index][0]], self.data[index][1], self.dict.tense2idx[self.target[index][1]]
