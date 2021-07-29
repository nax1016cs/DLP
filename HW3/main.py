from dataloader import *
from models import *
import numpy as np
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

def training(model, train_loader, test_loader, name):
    # best_record_file = name  + "_best_model"
    # for fname in os.listdir("check_point"):
    #     if fname[:len(best_record_file)] == best_record_file:
    #         best_test_acc = float(fname[len(best_record_file)+1:-3])
    #         print(best_record_file, best_test_acc)
    lr = 1e-4
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-3)
    best_test_acc = 0
    train_acc_record = []
    test_acc_record = []

    model.to(device=device)
    for epoch in range(epochs): 
        model.train()
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

        testing_accuracy, testing_loss = evaluating(model, test_loader, criterion, name)
        test_acc_record.append(testing_accuracy)


        print('epoch[\033[35m{:>4d}\033[00m/{:>4d}] {:.2f} sec(s) \033[32m Trian Acc:\033[00m {:.6f} Loss: {:.6f} | \033[33mTest Acc:\033[00m {:.6f} Loss: {:.6f}'.format(
            epoch+1, epochs, time.time() - epoch_start_time, training_accuracy, training_loss, testing_accuracy, testing_loss))
            
        if testing_accuracy > best_test_acc:
            best_test_acc = testing_accuracy
            save_checkpoint('check_point/{}_best_model' .format(name) + '.pt' , model)
    print("Best test accuracy of {}" .format(name) , best_test_acc)
    return np.array(train_acc_record), np.array(test_acc_record)

def evaluating(model, test_loader, criterion, name):
    test_acc, test_loss = 0.0, 0.0
    model.eval()
    y_pred = []
    y_true = []
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
            y_pred += [x.item() for x in torch.max(prediction, 1)[1]]
            y_true.extend(label.view(-1).detach().cpu().numpy()) 
    cf_matrix = confusion_matrix(y_true, y_pred)                              
    cf_matrix = cf_matrix/cf_matrix.sum(axis=1).reshape(-1, 1)
    plot_confusion_matrix(cf_matrix, name)
    return testing_accuracy, testing_loss

def plot_confusion_matrix(confusion_matrix, name):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{z:.2f}', ha='center', va='center')

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(name)

    plt.savefig("fig/" + name + ".png")

def save_checkpoint(path, model):
    state_dict = {"model_state_dict": model.state_dict(),}
    torch.save(state_dict, path)

def load_checkpoint(path, model, device):  
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])

    return model

def plot_accuracy(train_acc, test_acc, name, epoch):
    epoch = np.array([i for i in range(epoch)])
    plt.title('Accuracy', fontsize=18) 
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(epoch, train_acc*100, label = "train_" + name, linewidth=1)
    plt.plot(epoch, test_acc*100, label = "test_" + name, linewidth=2, linestyle="dashed")
    plt.legend(loc='upper left')
    plt.ylim([70, 105])

def main():
    train_loader = dataloader("train")
    test_loader = dataloader("test")
    resnet_18_pre = ResNet(18, pretrained=True)
    resnet_18_pre = load_checkpoint("check_point/" + "resnet_18_pre_best_model_0.7927402135231316.pt", resnet_18_pre, device)
    train_acc_record, test_acc_record= training(resnet_18_pre, train_loader, test_loader, "resnet_18_pre")
    plot_accuracy(train_acc_record, test_acc_record, "resnet_18_pre", 10)

    resnet_50_pre = ResNet(50, pretrained=True)
    resnet_50_pre = load_checkpoint("check_point/" + "resnet_50_pre_best_model_0.807117.pt", resnet_50_pre, device)
    train_acc_record, test_acc_record= training(resnet_50_pre, train_loader, test_loader, "resnet_50_pre")
    plot_accuracy(train_acc_record, test_acc_record, "resnet_50_pre", 10)

    resnet_18 = ResNet(18, pretrained=False)
    train_acc_record, test_acc_record= training(resnet_18, train_loader, test_loader ,"resnet_18")
    plot_accuracy(train_acc_record, test_acc_record, "resnet_18", 10)

    resnet_50 = ResNet(50, pretrained=False)
    train_acc_record, test_acc_record= training(resnet_50, train_loader, test_loader, "resnet_50")
    plot_accuracy(train_acc_record, test_acc_record, "resnet_50", 10)
    plt.savefig("fig/" + "Resnet" + ".png")
    plt.close()

def eval_best_model():
    train_loader = dataloader("train")
    test_loader = dataloader("test")
    models = [ "resnet_50_pre", "resnet_50", "resnet_18_pre", "resnet_18"]
    for name in models:
        num = int(name[7:9])
        model = ResNet(num, pretrained=True)
        model = load_checkpoint("check_point/" + name +"_best_model.pt", model, device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        testing_accuracy, _ = evaluating(model, test_loader, criterion, name)
        print(name + "_best_model: ", testing_accuracy)
    

if __name__ == '__main__':
    # main()
    eval_best_model()