from three_layer_nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
layer1 = []
def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y



class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,nn_layer,nn_layer_size,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)
        self.num_layer = nn_layer
        self.layer_size = nn_layer_size
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b2 = np.zeros((1, self.nn_hidden_dim))

    def feedforward(self, X, actFun):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.actFun(self.z1, self.actFun_type)
        a_layer = self.a1
        for i in range(self.num_layer):
            layer1[i] = layer(self.nn_input_dim, self.nn_hidden_dim, self.nn_output_dim, self.actFun_type, self.reg_lambda)
            a_layer = layer1[i].feedforward(a_layer,self.actFun_type)


    def backprop(self, X, y):
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        self.dW2 = (self.a1.T).dot(delta3)
        self.db2 = np.sum(delta3, axis=0, keepdims=True)
        delta1 = delta3
        for i in range(self.num_layer,0,step = -1):
            delta1 = layer1[i].backprop(X,y,delta1)
        self.dW1 = np.dot(X.T,delta1)
        self.db1 = np.sum(delta1, axis = 0)


    def calculate_loss(self, X, y):


    def fit_model(self, X, y, epsilon=0.005, num_passes=20000, print_loss=True):



class layer(NeuralNetwork):
    def __init__(self,id1,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,
                               seed=0)
        self.id = id1
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_hidden_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_hidden_dim))


    def feedforward(self, X, actFun):
        self.z = X.dot(self.W) + self.b1
        self.a = self.actFun(self.z, self.actFun_type)
        return self.a


    def backprop(self, X, delta, y):
        num_examples = len(X)
        self.delta_next = delta.dot(self.W.T) * self.diff_actFun(self.z, self.actFun_type)
        self.dw = np.dot(X.T,self.delta_next)
        self.db = np.sum(self.delta_next,axis=0)


        return self.delta_next

    def fit_model1(self,epsilon=0.005):
        self.dw +=self.reg_lambda*self.W
        self.W += -epsilon*self.dw
        self.b += -epsilon*self.db













