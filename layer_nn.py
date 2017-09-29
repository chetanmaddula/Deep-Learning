import numpy as np
class Layer(object):
    def __init__(self,id1,nn_input_dim ,nn_output_dim, nn_hidden_dim, actFun_type='tanh',input1 = False,output1 = False, reg_lambda=0.01, seed=0):

        self.id = id1
        self.isinput = input1
        self.isoutput = output1
        np.random.seed(seed)
        self.n_hidden = nn_hidden_dim
        if input1 == True:
            self.W = np.random.randn(nn_input_dim, self.n_hidden) / np.sqrt(nn_input_dim)
            self.b = np.zeros((1, self.n_hidden))

        elif output1 == True:
            self.W = np.random.randn(self.n_hidden, nn_output_dim) / np.sqrt(self.n_hidden)
            self.b = np.zeros((1, nn_output_dim))
        else:
            self.W = np.random.randn(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden)
            self.b = np.zeros((1, self.n_hidden))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda


    def feedforward(self, X, actFun):

        self.a_prev = X
        self.z = self.a_prev.dot(self.W) + self.b
        if self.isoutput:
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self.probs
        self.a = self.actFun(self.z, self.actFun_type)

        return self.a


    def backprop(self, X, delta, y,epsilon):

        num_examples = len(X)
        if self.isoutput:
            self.delta = self.probs
            self.delta[range(num_examples),y] -= 1
            self.dw = np.dot(self.a_prev.T, self.delta)
            self.db = np.sum(self.delta, axis=0)
            self.delta_next = self.delta.dot(self.W.T)
            return self.delta_next

        elif self.isinput:

            self.delta = delta*self.diff_actFun(self.z, self.actFun_type)
            self.dw = np.dot(self.a_prev.T,self.delta)
            self.db = np.sum(self.delta,axis=0)
        else:
            self.delta = delta * self.diff_actFun(self.z, self.actFun_type)
            self.dw = np.dot(self.a_prev.T, self.delta)
            self.db = np.sum(self.delta, axis=0)
            self.delta_next = self.delta.dot(self.W.T)
            return self.delta_next

        self.dw += self.reg_lambda * self.W
        self.W += -epsilon * self.dw
        self.b += -epsilon * self.db



    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        if type == 'tanh':
            activations = np.tanh(z)
        elif type == 'sigmoid':
            activations = 1 / (1 + np.exp(-z))
        else:
            activations = np.maximum(z, 0)
        return activations

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''
        if type == 'tanh':
            dL = 1 - self.a ** 2
        elif type == 'sigmoid':
            sig_fn = self.actFun(z, type)
            dL = self.a * (1 - self.a)
        else:
            dL = 1 * (z > 0)

        # YOU IMPLEMENT YOUR diff_actFun HERE

        return dL