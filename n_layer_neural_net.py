from three_layer_nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,nn_layer,nn_layer_size,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)
        self.num_layer = nn_layer
        self.layer_size = nn_layer_size

    def feedforward(self, X, actFun):

    def backprop(self, X, y):
        delta1 = num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        dW2 = (self.a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta1 = delta3.dot(self.W2.T) * self.diff_actFun(self.z1, self.actFun_type)
        for i in range(self.nn_layer):
            layer1 = layer(self.nn_input_dim, self.nn_hidden_dim, self.nn_output_dim, self.actFun_type, self.reg_lambda)
            delta1 = layer1.backprop(X,y,delta1)


    def calculate_loss(self, X, y):

    def fit_model(self, X, y, epsilon=0.005, num_passes=20000, print_loss=True):

class layer(NeuralNetwork):
    def __init__(self,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,
                               seed=0)
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))


    def feedforward(self, X, actFun):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.actFun(self.z1, self.actFun_type)


    def backprop(self, X, delta, y):
        num_examples = len(X)

        self.dw = np.dot(X.T,delta)
        self.db = np.sum(self.delta,axis=0)
        self.delta_next = self.delta.dot(self.W1.T)*self.diff_actFun(self.z1,self.actFun_type)

        return self.delta_next










