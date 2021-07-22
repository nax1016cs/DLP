from dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EEGNET(nn.Module):
    def __init__(self, activation, dropout=0.5):
        super(EEGNET, self).__init__()
        self.name = "EEGNET_" + activation
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
# 0.1 0.1 0.8527
# 0.1 0.05 0.856
# 0.1 0.0005 0.857

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.seperableConv(out)
        out = out.view(out.shape[0],-1)
        out = self.classify(out)
        return out

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.name = "DeepConvNet_" + activation
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
            nn.Dropout(0.5)
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
    model.load_state_dict(state_dict["model_state_dict"])

    return model

def plot_accuracy(acc):
    pass

def training(model, train_loader, test_loader):
    print(device)
    lr = 1e-3
    epochs = 500
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.012)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0.0
    train_acc_record, train_loss_record = [], []
    model.to(device=device)
    # model.train()
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
            train_loss_record.append(training_loss)

        testing_accuracy, testing_loss = evaluating(model, test_loader, criterion)


        if (epoch+1)%5 == 0 or epoch == 0:
            print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m Trian Acc:\033[00m {:.6f} Loss: {:.6f} | \033[33mTest Acc:\033[00m {:.6f} Loss: {:.6f}'.format(
                epoch+1, epochs, time.time() - epoch_start_time, training_accuracy, training_loss, testing_accuracy, testing_loss))
            
        if testing_accuracy > best_test_acc:
            best_test_acc = testing_accuracy
            save_checkpoint('check_point/{}_best_model.pt'.format(model.name), model)
    print("Best test accuracy of {}" .format(model.name) , best_test_acc)

def evaluating(model, test_loader, criterion):
    test_acc_record, test_loss_record = [], []
    test_acc, test_loss = 0.0, 0.0
    # model.eval()
    # with torch.no_grad():
    #     loss, correct = 0, 0
    #     for idx, data in enumerate(test_loader):
    #         x, y = data
    #         inputs = x.to(device, dtype=torch.float)
    #         labels = y.to(device, dtype=torch.long)

    #         outputs = model(inputs)
    #         loss += criterion(outputs, labels)

    #         correct += (
    #             torch.max(outputs, 1)[1] == labels.long().view(-1)
    #         ).sum().item()
    # return correct  / len(test_loader.dataset), loss.item()/ len(test_loader.dataset)
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
            test_acc_record.append(testing_accuracy)
            test_loss_record.append(testing_loss)
    return testing_accuracy, testing_loss

train_loader, test_loader = dataloader()
# activation_func = ["elu", "leaky_relu", "relu"]
activation_func = ["relu"]
for func in activation_func:
    eegnet = EEGNET(func, dropout=0.15)
    training(eegnet, train_loader, test_loader)
    # deepconvnet = DeepConvNet(func)
    # training(deepconvnet, train_loader, test_loader)

# model = DeepConvNet('relu')
# deepconvnet = load_checkpoint('check_point/DeepConvNet_relu_best_model.pt', model, device)
# model.to(device=device)
# evaluating(deepconvnet, test_loader, nn.CrossEntropyLoss())