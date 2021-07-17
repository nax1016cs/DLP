import numpy as np
import matplotlib.pyplot as plt
import math
import time
def sigmoid(x):
    return 1/(1+math.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def gen_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def gen_xor(n=31):
    inputs = []
    labels = []
    ratio = 1 / (n-1)
    for i in range(n):
        inputs.append([ratio*i, ratio*i])
        labels.append(0)

        if ratio*i == 0.5:
            continue

        inputs.append([ratio*i, 1-ratio*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(2*n-1,1) 

def show_result(x, y, predictions, name):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18) 
    for i in range(x.shape[0]):
        if predictions[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig('result/' + name + "_result" + ".png")
    plt.close()

def plot_training_curve(loss_per_epoch, mode):

    plt.title(mode + ' learning curve', fontsize=18) 
    epoch, loss = list(map(list,zip(*loss_per_epoch)))[0], list(map(list,zip(*loss_per_epoch)))[1]
    fig = plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_ylim([0,7])
    ax1.set_title('Learning curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epoch, loss, color='tab:red', label='Loss')
    plt.savefig('result/' + mode + "_loss" + ".png")
    plt.close()

class neuron:
    def __init__(self, input_size):
        self.weight = np.array([np.random.randn() for i in range(input_size)])
        self.z = 0
        self.a = 0
        self.input_data = 0
        self.partial_c_z = 0

    def forwarding(self, input_data):
        self.input_data = input_data
        self.z = self.weight.T@input_data
        self.a = sigmoid(self.z)

    def update(self, lr):
        for i in range(len(self.weight)):
            self.weight[i] -= lr * self.input_data[i] * self.partial_c_z

class layer:
    def __init__(self, input_size, size):
        self.n = [neuron(input_size) for i in range(size)]

class NN:
    def __init__(self, name, size = 5, input_size = 2, lr = 1e-2):
        self.name = name
        self.lr = lr
        self.size = [size, size, 1]
        self.layers = [layer(2,size), layer(size, size), layer(size, 1)]

    def forward(self, input_data):
        for layer in range(3):
            for i in range(self.size[layer]):
                self.layers[layer].n[i].forwarding(np.array(input_data))
            input_data = []
            for i in range(self.size[layer]):
                input_data.append(self.layers[layer].n[i].a)
    
    def backward(self, y):
        
        for layer in reversed(range(3)):
            for i in range(self.size[layer]):
                if layer == 2:
                    self.layers[layer].n[0].partial_c_z = 2 * (self.layers[layer].n[0].a - y)
                else: 
                    for j in range(self.size[layer+1]):
                        self.layers[layer].n[i].partial_c_z += \
                            self.layers[layer+1].n[j].partial_c_z * self.layers[layer+1].n[j].weight[i]
                self.layers[layer].n[i].partial_c_z *= derivative_sigmoid(self.layers[layer].n[i].z)
        for layer in reversed(range(3)):
            for i in range(self.size[layer]):
                self.layers[layer].n[0].update(self.lr)
    
    def training(self, x, y, epochs=5000):
        loss_per_epoch = []
        for epoch in range(epochs):
            loss = 0

            for j in range(len(x)):
                self.forward(x[j])
                prediction = self.layers[2].n[0].a
                loss += ( prediction - y[j] ) ** 2
                self.backward(y[j])
            if epoch % 100 == 0:
                print("Epoch ", epoch , " loss: ", loss)
                loss_per_epoch.append([epoch,loss])
            # if loss < 0.1:
            #     break
            # acc = 0
            # for i in range(len(x)):
            #     self.forward(x[i])
            #     prediction = self.layers[2].n[0].a
            #     if prediction > 0.5 and y[i] == 1 or prediction < 0.5 and y[i] ==0 :
            #         acc += 1
            # if acc == len(x):
            #     break

        plot_training_curve(loss_per_epoch, self.name)
            
    def testing(self, x, y):
        predictions = []
        acc = 0
        for i in range(len(x)):
            self.forward(x[i])
            prediction = self.layers[2].n[0].a
            predictions.append(prediction)
            if prediction > 0.5 and y[i] == 1 or prediction < 0.5 and y[i] ==0 :
                acc += 1
            print(prediction)
        print("Accuracy: ", (acc / len(x))*100, "%")
        show_result(x, y, predictions, self.name )

    def save_weight(self, name):
        weights = []
        for layer in self.layers:
            for neuron in layer.n:
                weights.append(neuron.weight)
        np.save(name, np.array(weights))

    def load_weight(self, name):
        weight = np.load(name, allow_pickle=True)
        i = 0
        for layer in self.layers:
            for neuron in layer.n:
                neuron.weight = weight[i]
                i += 1

linear_x, linear_y = np.load('data/linear_x.npy'), np.load('data/linear_y.npy')
xor_x, xor_y = np.load('data/xor_x.npy'), np.load('data/xor_y.npy')




size = 8

linear_NN = NN("linear_" + str(size) + "_" + str(0.01), size,)
# linear_NN.training(linear_x, linear_y,  epochs = 1000)
# linear_NN.save_weight("result/linear_8_final_weight")
linear_NN.load_weight("result/linear_8_final_weight.npy")
linear_NN.testing(linear_x, linear_y)


xor_NN = NN("xor_" + str(size) + "_" + str(0.05), size, lr = 0.05)
# xor_NN.training(xor_x, xor_y,  epochs = 3000)

# xor_NN.save_weight("result/xor_8_final_weight")
xor_NN.load_weight("result/xor_8_final_weight.npy")
xor_NN.testing(xor_x, xor_y)



