import random
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from utils import *
from record import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 28
conditoin_output_size = 8
latent_size = 32
teacher_forcing_ratio = 1
empty_input_ratio = 0.1
KLD_weight = 0
lr = 0.05

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hn, cn):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hn,cn) = self.lstm(output, (hn, cn))
        return output, hn, cn

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, intput_size, hidden_size ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(intput_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, intput_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hn, cn):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hn, cn) = self.lstm(output, (hn, cn))
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hn, cn

class CVAE(nn.Module):
    def __init__(self, input_size, hidden_size, condition_input_size , conditoin_output_size,  latent_size):
        super(CVAE, self).__init__()
        self.condition_embedding = nn.Embedding(condition_input_size, conditoin_output_size)
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size + conditoin_output_size, hidden_size)
        self.latent2cell = nn.Linear(latent_size + conditoin_output_size, hidden_size)
        self.decoder = DecoderRNN(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.conditoin_output_size = conditoin_output_size
        self.condition_input_size = condition_input_size
        self.latent_size = latent_size
    
    def initHidden(self, input_condition):
        h0 = torch.zeros(self.hidden_size - self.conditoin_output_size, device=device).view(1, 1, -1)
        c0 = torch.zeros(self.hidden_size, device=device).view(1, 1, -1)
        h0 = torch.cat((h0, self.condition(input_condition)), dim=-1)
        return h0, c0
    
    def sampling(self):
        return torch.normal(
            torch.FloatTensor([0] * (self.latent_size)),
            torch.FloatTensor([1] * (self.latent_size))
        ).to(device).view(1, 1, -1)

    def KL_loss(self, mean, log_var):
        return torch.mean(0.5 * (-log_var + (mean ** 2) + torch.exp(log_var) - 1))

    def condition(self, c):
        return self.condition_embedding(torch.LongTensor([c]).to(device)).view(1, 1, -1)

    def forward(self, input, input_condition, target, target_condtion, teacher_forcing=True):
        hn, cn = self.initHidden(input_condition)
        for i in range(input.size(0)):
            _, hn, cn = self.encoder(input[i], hn,cn )
        mean = self.mean(hn)
        log_var = self.logvar(hn)
        latent = self.sampling() * torch.exp(log_var / 2) + mean
        hn = self.latent2hidden(torch.cat((latent, self.condition(target_condtion)), dim=-1).reshape(-1)).view(1, 1, -1)
        cn = self.latent2cell(torch.cat((latent, self.condition(target_condtion)), dim=2).reshape(-1)).view(1, 1, -1)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        prediction = []
        for i in range(MAX_LENGTH):
            output, hn, cn = self.decoder(decoder_input, hn, cn)
            if teacher_forcing:
                if i == target.size(0):
                    break
                prediction.append(output)
                decoder_input = target[i]
            else:
                prediction.append(output)
                topv, topi = output.topk(1)
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_token:
                    break
        return torch.stack(prediction), mean, log_var
    
    def get_teacher_forcing_ratio(self, epoch):
        teacher_forcing_ratio = 1.-(1./(500-1))*(epoch)
        return teacher_forcing_ratio

    def get_kl_weight(self, epoch, type_):

        if type_ == 'monotonic':
            return 0.1*epoch if epoch < 10 else 1 

        elif type_ == 'cyclical': 
            return 1 if epoch % 16 > 10  else (epoch % 10) * 0.1


    def Train(self, train_dataset, optimizer, epoch):
        self.train()
        criterion = nn.CrossEntropyLoss()
        total_cross_entropy_loss, total_kld_loss = 0, 0
        idx_pool = list(range(len(train_dataset)))
        random.shuffle(idx_pool)
        for i in idx_pool:
            optimizer.zero_grad()
            data, condition = train_dataset[i]
            target = train_dataset.dict.encode(data).cuda()
            teacher_forcing_ratio = self.get_teacher_forcing_ratio(epoch)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            output, mean, log_var = self.forward(target, condition, target, condition, use_teacher_forcing)
            cross_entropy_loss = 0

            for j in range(min(len(target), len(output))):
                cross_entropy_loss += criterion(output[j], target[j])
            cross_entropy_loss /= len(output)

            kld_loss = self.KL_loss(mean, log_var)
            loss = cross_entropy_loss + kld_loss * self.get_kl_weight(epoch,'monotonic')
            total_cross_entropy_loss += cross_entropy_loss.item()
            total_kld_loss += kld_loss.item()
            loss.backward()
            optimizer.step()
        return total_cross_entropy_loss / len(train_dataset), total_kld_loss / len(train_dataset), self.get_kl_weight(epoch,'monotonic'), teacher_forcing_ratio
    
    def Test(self, test_dataset):
        self.eval()
        bleu_acc = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                input_, input_condition, target, target_condition = test_dataset[i]
                input_data = test_dataset.dict.encode(input_)
                output, mean, logg_var = self.forward(input_data, input_condition, None, target_condition, False)
                output = test_dataset.dict.decode(output.argmax(dim=2).view(-1, 1))
                bleu_acc += compute_bleu(output[:-3], target)
                print (f"input: {input_:20}, target: {target:20}, output: {output[:-3]:20}")
            return bleu_acc / len(test_dataset)

    def sample(self, dataset):
        self.eval()
        z = self.sampling()
        word = []
        with torch.no_grad():
            for tense in range(4):
                hn =  self.latent2hidden(torch.cat((z, self.condition(tense)), dim=2).reshape(-1)).view(1, 1, -1)
                cn =  self.latent2cell(torch.cat((z, self.condition(tense)), dim=2).reshape(-1)).view(1, 1, -1)
                
                input = torch.tensor([[SOS_token]], device=device)
                prediction = []
                for i in range(MAX_LENGTH):
                    output, hn, cn = self.decoder(input, hn, cn)
                    prediction.append(output)
                    topv, topi = output.topk(1)
                    input = topi.squeeze().detach()
                    if input.item() == EOS_token:
                        break

                output = torch.stack(prediction)
                output = dataset.dict.decode(output.argmax(dim=2).view(-1, 1))
                word.append(output[:-3])
        return word
    

