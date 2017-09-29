from three_layer_nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math
from layer_nn import Layer
from sklearn import datasets

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y



class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,nn_layer ,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)
        self.num_layer = nn_layer
        np.random.seed(seed)
        self.nn_hidden_dim = nn_hidden_dim
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))
        self.layers1 = []
        self.actFun_type = actFun_type

        for i in range(self.num_layer):
            if i == 0:
                layer1obj = Layer(i,self.nn_input_dim,self.nn_output_dim,self.nn_hidden_dim, self.actFun_type, input1= True,reg_lambda = self.reg_lambda)
            elif i == self.num_layer -1:
                layer1obj = Layer(i,self.nn_input_dim,self.nn_output_dim, self.nn_hidden_dim, self.actFun_type, output1=True, reg_lambda=self.reg_lambda)
            else:
                layer1obj = Layer(i,self.nn_input_dim,self.nn_output_dim, self.nn_hidden_dim, self.actFun_type, reg_lambda=self.reg_lambda)
            self.layers1.append(layer1obj)

    def feedforward(self, X, actFun):

        a_layer = X
        for i in range(self.num_layer-1):
            a_layer = self.layers1[i].feedforward(a_layer,self.actFun_type)
        self.probs = self.layers1[self.num_layer-1].feedforward(a_layer,self.actFun_type)

        return None


    def backprop(self, X, y,epsilon = 0.01):
        delta1 = self.probs
        for i in range(self.num_layer-1,-1,-1):
            delta1 = self.layers1[i].backprop(X,delta1,y,epsilon)

        return None


    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss
        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        corect_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        w_sum = 0
        for k in range(self.num_layer):
            w_sum += np.sum(np.square(self.layers1[k].W))

        data_loss += self.reg_lambda / 2 * (w_sum)
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.005, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X,y,epsilon)

            # Add regularization terms (b1 and b2 don't have regularization terms)


            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i,%d: %f" % (i, self.nn_hidden_dim, self.calculate_loss(X, y)))






def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='tanh')
    # model.fit_model(X, y)
    # model.visualize_decision_boundary(X, y)
    plt.figure(figsize=(16, 32))
    hidden_lay = [2]
        # plt.subplot(5,2,i+1)
        # plt.title('Hidden layer with size %d' % nn_hidden_dim1)
    model = DeepNeuralNetwork(nn_layer=10,nn_input_dim=2, nn_hidden_dim=5, nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()










