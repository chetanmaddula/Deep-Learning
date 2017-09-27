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


    def backprop(self, X, y):
        num_examples = len(X)
        self.delta =
        self.dw = np.dot(X.T,self.delta)
        self.db = np.sum(self.delta,axis=0)










