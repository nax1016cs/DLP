import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

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

def gen_xor():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1) 

def show_result(x, y, predictions):
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
        if predictions[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

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
            self.weight[i] += lr * self.input_data[i] * self.partial_c_z
            # print("weights: ", self.weight[i])

class layer:
    def __init__(self, input_size, size):
        self.n = [neuron(input_size) for i in range(size)]

class NN:
    def __init__(self, size = 5, input_size = 2, lr = 1e-3):
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
                # print("layer: ", layer, "n: ", i)
                # print(self.layers[layer].n[i].a)
    
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
        for epoch in range(epochs):
            loss = 0
            for j in range(len(x)):
                self.forward(x[j])
                prediction = self.layers[2].n[0].a
                # print(prediction)
                loss += ( prediction - y[j] ) ** 2
                self.backward(y[j])
            if epoch % 50 == 0:
                print("Epoch ", epoch , " loss: ", loss)
            
    def testing(self, x, y):
        predictions = []
        for i in range(len(x)):
            self.forward(x[i])
            prediction = self.layers[2].n[0].a
            predictions.append(prediction)
        show_result(x, y, predictions)


linear_x, linear_y = np.load('linear_x.npy'), np.load('linear_y.npy')
xor_x, xor_y = np.load('xor_x.npy'), np.load('xor_y.npy')
a = NN(3)
a.training(linear_x, linear_y)
a.testing(linear_x, linear_y)

