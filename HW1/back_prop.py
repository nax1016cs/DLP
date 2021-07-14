import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)

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

def show_result(x, y, predict_y):
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
        if predict_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

ct = 0


class neuron:
    def __init__(self, input_size):
        # self.weight = [np.random.randn() for i in range(input_size)]
        global ct
        self.weight = np.array([ct+i for i in range(input_size)])
        ct = ct + input_size
        self.z = 0
        self.a = 0
        self.partial_c_partial_z = 0

    def forwarding(self, input_data):
        print(self.weight)
        self.z = self.weight.T@input_data
        self.a = sigmoid(self.z)

    def update(self):
        pass

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
                input_data.append(self.layers[layer].n[i].z)
                print("layer: ", layer, "n: ", i)
                print(self.layers[layer].n[i].z)
    


linear_x, linear_y = np.load('linear_x.npy'), np.load('linear_y.npy')
xor_x, xor_y = np.load('xor_x.npy'), np.load('xor_y.npy')


a = NN(3)
a.forward(np.array([0,1]))

# show_result(linear_x, linear_y, linear_y)

