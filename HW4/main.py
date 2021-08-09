from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from os import system
import os
from dataloader import * 
from models import *
from record import *


"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict(),}
    torch.save(state_dict, path)

def load_checkpoint(path, model, device):  
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def training(model, train_dataset, epochs=100):
    best_acc = float(os.listdir("check_point")[0].split(".pt")[0])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    r = record()

    for epoch in range(100):
        epoch_start_time = time.time()
        cross_entropy_loss, kld_loss, kld_weight, teacher_forcing_ratio = model.train(train_dataset, optimizer, epoch)
        bleu_acc = model.test(test_dataset)
        gaussian_acc = gaussian_scoring(model, train_dataset)
        r.add_record(cross_entropy_loss, kld_loss, teacher_forcing_ratio, kld_weight, bleu_acc, gaussian_acc)
        if bleu_acc > best_acc:
            save_checkpoint("check_point/"+ str(bleu_acc), model)
            best_acc = bleu_acc
        print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m Trian Acc:\033[00m {:.6f} KLD_loss: {:.6f} | \033[33mCR_loss:\033[00m {:.6f}'.format(
            epoch+1, epochs, time.time() - epoch_start_time, bleu_acc, kld_loss, cross_entropy_loss))
    r.plot()
        
def gaussian_scoring(model, train_dataset):
    words = []
    torch.random.manual_seed(460)
    for i in range(100):
        words.append(model.sample(train_dataset))
    return Gaussian_score(words)

train_dataset = TrainDataset('data/train.txt')
test_dataset = TestDataset('data/test.txt')

model = CVAE(len(train_dataset), hidden_size,  4 , 8 ,latent_size)
# model = load_checkpoint("check_point/0.5922075628542363", model, device)
model.to(device)
training(model, train_dataset)
