from three_layer_nn import NeuralNetwork


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,nn_layer,nn_layer_size,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        NeuralNetwork.__init__(self,nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)
        self.num_layer = nn_layer
        self.layer_size = nn_layer_size

        def feedforward(self, X, actFun):

        def backprop(self, X, y):

        def calculate_loss(self, X, y):

        def fit_model(self, X, y, epsilon=0.005, num_passes=20000, print_loss=True):

class layer()



