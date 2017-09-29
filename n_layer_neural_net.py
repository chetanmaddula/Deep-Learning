from three_layer_nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math
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
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b2 = np.zeros((1, self.nn_hidden_dim))
        self.layers1 = []

        for i in range(self.num_layer):
            layer1obj = layer(self.nn_input_dim, self.nn_hidden_dim, self.nn_output_dim, self.actFun_type, self.reg_lambda)
            self.layers1.append(layer1obj)

    def feedforward(self, X, actFun):

        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.actFun(self.z1, self.actFun_type)
        a_layer = self.a1

        for i in range(self.num_layer):
            a_layer = self.layers1[i].feedforward(a_layer,self.actFun_type)

        self.z2 = a_layer.dot(self.W2) + self.b2

        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None


    def backprop(self, X, y,epsilon = 0.01):
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1

        self.dW2 = (self.a1.T).dot(delta3)
        self.db2 = np.sum(delta3, axis=0, keepdims=True)
        delta1 = delta3.dot(self.W2.T)

        for i in range(self.num_layer,0,step = -1):
            delta1 = self.layers1[i].backprop(X,delta1,y,epsilon)

        delta1 = delta1*self.diff_actFun(self.z1,self.actFun_type)
        self.dW1 = np.dot(X.T,delta1)
        self.db1 = np.sum(delta1, axis = 0)

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
            self.dW2 += self.reg_lambda * self.W2 / len(X)
            self.dW1 += self.reg_lambda * self.W1 / len(X)

            # Gradient descent parameter update
            self.W1 += -epsilon * self.dW1
            self.b1 += -epsilon * self.db1
            self.W2 += -epsilon * self.dW2
            self.b2 += -epsilon * self.db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i,%d: %f" % (i, self.nn_hidden_dim, self.calculate_loss(X, y)))



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


    def backprop(self, X, delta, y,epsilon):
        num_examples = len(X)
        self.delta_next = delta*self.diff_actFun(self.z, self.actFun_type)
        self.dw = np.dot(X.T,self.delta_next)
        self.db = np.sum(self.delta_next,axis=0)
        self.delta_next = self.delta_next.dot(self.W.T)
        self.dw += self.reg_lambda * self.W
        self.W += -epsilon * self.dw
        self.b += -epsilon * self.db

        return self.delta_next


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
    model = DeepNeuralNetwork(nn_layer=7,nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()










