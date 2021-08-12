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


def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict(),}
    torch.save(state_dict, path)

def load_checkpoint(path, model, device):  
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    return model

def training(model, train_dataset, test_dataset, epochs=100):
    best_acc = 0.9
    optimizer = optim.SGD(model.parameters(), lr=lr)
    r = record()

    for epoch in range(100):
        epoch_start_time = time.time()
        cross_entropy_loss, kld_loss, kld_weight, teacher_forcing_ratio = model.Train(train_dataset, optimizer, epoch)
        bleu_acc = model.Test(test_dataset)
        gaussian_acc = gaussian_scoring(model, train_dataset)
        r.add_record(cross_entropy_loss, kld_loss, teacher_forcing_ratio, kld_weight, bleu_acc, gaussian_acc)
        if (bleu_acc + gaussian_acc ) > best_acc : 
            save_checkpoint("check_point/"+ str(bleu_acc) + '_' + str(gaussian_acc), model)
            best_acc = (bleu_acc+gaussian_acc)
        print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m BLEU-4:\033[00m {:.6f} \033[32m Gaussian:\033[00m {:.6f} KLD_loss: {:.6f} | \033[33mCR_loss:\033[00m {:.6f} \033[33mTF:\033[00m {:.6f} \033[33mKL_weight:\033[00m {:.6f} '.format(
            epoch+1, epochs, time.time() - epoch_start_time, bleu_acc,gaussian_acc, kld_loss, cross_entropy_loss, teacher_forcing_ratio, kld_weight))
    r.plot()
        

def evaluate(model, train_dataset, test_dataset):
    torch.random.manual_seed(243)
    bleu_acc = model.Test(test_dataset)
    print("BLEU4 score: ", bleu_acc)
    gaussian_acc = gaussian_scoring(model, train_dataset)
    print("Gaussian score: ", gaussian_acc)

def gaussian_scoring(model, train_dataset):
    train_words = []
    for i in range(int(len(train_dataset)/4)):
        word = []
        for j in range(4):
            word.append(train_dataset[i*4+j][0])
        train_words.append(word)
    words = []
    torch.random.manual_seed(460)
    for i in range(100):
        gen_word = model.sample(train_dataset)
        words.append(gen_word)
        if gen_word in train_words:
            print("--->", end='')
        print("\t", gen_word)
    return Gaussian_score(words)



train_dataset = TrainDataset('data/train.txt')
test_dataset = TestDataset('data/test.txt')

model = CVAE(len(train_dataset), hidden_size,  4 , 8 ,latent_size)
model = load_checkpoint("check_point/1.0_0.33", model, device)
model.to(device)
evaluate(model, train_dataset, test_dataset)
# training(model, train_dataset, test_dataset)
