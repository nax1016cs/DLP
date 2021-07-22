from dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EEGNET(nn.Module):
    def __init__(self, activation, dropout=0.25):
        super(EEGNET, self).__init__()
        self.name = "EEGNET_" + activation + "_" + str(dropout)
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU() 
        elif activation == "relu":
            self.activation = nn.ReLU()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            self.activation, 
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(dropout)
        )
        self.seperableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            self.activation, 
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(dropout)
        )
        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias=True)
        )

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.seperableConv(out)
        out = out.view(out.shape[0],-1)
        out = self.classify(out)
        return out

class DeepConvNet(nn.Module):
    def __init__(self, activation, dropout=0.5):
        super(DeepConvNet, self).__init__()
        self.dropout = dropout
        self.name = "DeepConvNet_" + activation + "_" + str(dropout)
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU() 
        elif activation == "relu":
            self.activation = nn.ReLU()
        conv2d_kernel_size = [(2,1), (1,5), (1,5), (1,5)]

        self.block0 = nn.Conv2d(1,25, kernel_size=(1,5))
        for i in range(1,5):
            self.make_block(conv2d_kernel_size[i-1], (1,2), i)
        self.classify = nn.Sequential(
            nn.Linear(8600, 2, bias=True)
        )
    
    def make_block(self, conv2d_size, pool_size, i):
        filters = [25,25,50,100,200]
        setattr(self,'block'+str(i),nn.Sequential(
            nn.Conv2d(filters[i-1],filters[i], conv2d_size),
            nn.BatchNorm2d(filters[i], eps=1e-5, momentum=0.1),
            self.activation,
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout(self.dropout)
        ))
        
    def forward(self, x):

        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.shape[0],-1)
        out = self.classify(out)
        return out

        

def dataloader():
    train_data, train_label, test_data, test_label = read_bci_data()
    training_set = TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
    testing_set = TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
    train_loader = DataLoader(training_set, batch_size=1080, shuffle=True)
    test_loader = DataLoader(testing_set, batch_size=1080, shuffle=False)
    return train_loader, test_loader


def save_checkpoint(path, model):
    
    state_dict = {"model_state_dict": model.state_dict(),}
    torch.save(state_dict, path)

def load_checkpoint(path, model, device):  
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"] , strict=True)

    return model

def plot_accuracy(acc, name):
    epoch = np.array([i for i in range(300)])
    plt.title('Accuracy', fontsize=18) 
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(epoch, acc*100, label = name, linewidth=0.5)
    plt.legend(loc='lower right')
    plt.ylim([50, 105])



def training(model, train_loader, test_loader):
    print(device)
    lr = 1e-3
    epochs = 300
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.012)
    best_test_acc = 0.0
    train_acc_record = []
    test_acc_record = []

    model.to(device=device)
    model.train()
    for epoch in range(epochs): 
        epoch_start_time = time.time()
        train_acc, train_loss = 0.0, 0.0
        for training_data in train_loader:
            data = training_data[0].to(device, dtype=torch.float)
            label = training_data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            prediction = model(data)
            batch_loss = criterion(prediction, label)
            batch_loss.backward()
            optimizer.step()
            train_acc += prediction.max(dim=1)[1].eq(label).sum().item()
            train_loss += batch_loss.item()
            
            training_accuracy = train_acc / len(train_loader.dataset)
            training_loss = train_loss / len(train_loader.dataset)
            
            train_acc_record.append(training_accuracy)

            testing_accuracy, testing_loss = evaluating(model, test_loader, criterion)
            test_acc_record.append(testing_accuracy)


        if (epoch+1)%5 == 0 or epoch == 0:
            print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m Trian Acc:\033[00m {:.6f} Loss: {:.6f} | \033[33mTest Acc:\033[00m {:.6f} Loss: {:.6f}'.format(
                epoch+1, epochs, time.time() - epoch_start_time, training_accuracy, training_loss, testing_accuracy, testing_loss))
            
        if testing_accuracy > best_test_acc:
            best_test_acc = testing_accuracy
            save_checkpoint('check_point/{}_best_model.pt'.format(model.name), model)
    print("Best test accuracy of {}" .format(model.name) , best_test_acc)
    return np.array(train_acc_record), np.array(test_acc_record)

def evaluating(model, test_loader, criterion):
    test_acc, test_loss = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for testing_data in test_loader:
            data = testing_data[0].to(device, dtype=torch.float)
            label = testing_data[1].to(device, dtype=torch.long)
            prediction = model(data)
            batch_loss = criterion(prediction, label)
            test_acc += prediction.max(dim=1)[1].eq(label).sum().item()
            test_loss += batch_loss.item()
            testing_accuracy = test_acc / len(test_loader.dataset)
            testing_loss = test_loss / len(test_loader.dataset)
    return testing_accuracy, testing_loss

droupt_rate = [0.15]
train_loader, test_loader = dataloader()
# activation_func = ["elu", "leaky_relu", "relu"]
activation_func = ["relu"]
for dropout in droupt_rate:
    for func in activation_func:
        eegnet = EEGNET(func, dropout=dropout)
        train_acc_record, test_acc_record = training(eegnet, train_loader, test_loader)
        np.save("Accuracy/" + "EEG_train_"+ str(dropout) + "_" + func, train_acc_record)
        plot_accuracy(train_acc_record, "EEG_train_"+ str(dropout) + "_" + func)
        np.save("Accuracy/" + "EEG_test_" + str(dropout) + "_"+ func, test_acc_record)
        plot_accuracy(test_acc_record, "EEG_test_"+ str(dropout) + "_" + func)
plt.savefig("fig/" + "EEG" + ".png")
plt.close()

# for dropout in droupt_rate:
#     for func in activation_func:
#         deepconvnet = DeepConvNet(func, dropout=dropout)
#         train_acc_record, test_acc_record = training(deepconvnet, train_loader, test_loader)
#         np.save("Accuracy/" + "DEEP_train_" + str(dropout) + "_"+ func, train_acc_record)
#         plot_accuracy(train_acc_record, "DEEP_train_"+ str(dropout) + "_" + func)
#         np.save("Accuracy/" + "DEEP_test_"  + str(dropout) + "_"+  func, test_acc_record)
#         plot_accuracy(test_acc_record, "DEEP_test_"+ str(dropout) + "_" + func)
# plt.savefig("fig/" + "DEEP" + ".png")
# plt.close()
# train_acc_record = np.load("Accuracy/" + "DEEP_train_" + str(0.5) + "_relu.npy")
# test_acc_record = np.load("Accuracy/" + "DEEP_train_" + str(0.5) + "_elu.npy")
# print(test_acc_record.shape)
# print(train_acc_record.shape)
# plot_accuracy(train_acc_record, "DEEP_train_" + str(0.5) + "_relu")
# plot_accuracy(test_acc_record, "DEEP_train_" + str(0.5) + "_elu")
