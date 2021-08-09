import numpy as np
import matplotlib.pyplot as plt

class record:
    def __init__(self):
        self.cross_entropy = []
        self.kld = []
        self.teacher_forcing = []
        self.kld_weight = []
        self.bleu_score = []
        self.gaussian_score = []
    
    def add_record(self, ce_loss, kld_loss, teacher_forcing_ratio, kld_weight_, bleu, gaussian):
        self.cross_entropy.append(ce_loss)
        self.kld.append(kld_loss)
        self.teacher_forcing.append(teacher_forcing_ratio)
        self.kld_weight.append(kld_weight_)
        self.bleu_score.append(bleu)
        self.gaussian_score.append(gaussian)
    
    def plot(self):
        epoch = np.array([i for i in range(len(self.cross_entropy))])
        plt.title('Training information', fontsize=18) 
        plt.xlabel("Epoch")
        plt.ylabel("loss/ratio/accuracy")
        plt.plot(epoch, self.bleu_score, label = "BLEU4-score" , linewidth = 1 )
        plt.plot(epoch, self.gaussian_score, label = "gaussian_score" , linewidth = 1 )
        plt.plot(epoch, self.kld, label = "KLD" , linewidth = 1 )
        plt.plot(epoch, self.cross_entropy, label = "CrossEntropy" , linewidth = 1 )
        plt.plot(epoch, self.kld_weight, label = "kld_weight" , linewidth=2, linestyle="dashed")
        plt.plot(epoch, self.teacher_forcing, label = "teacher ratio" , linewidth=2, linestyle="dashed")
        plt.legend(loc='upper left')
        plt.savefig("fig/result.png")
