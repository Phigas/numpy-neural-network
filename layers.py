import numpy as np
from typing import Callable

class Function:
    def __init__(self, run, derivate) -> None:
        self.run = run
        self.derivate = derivate

class Layer:
    def __init__(self, input_size, node_nr, activation) -> None:
        self.n_in = input_size[0] # input nodes
        self.m = input_size[1] # batch size
        self.n = node_nr
        
        # needs initialization
        self.weights = (np.random.rand(self.n, self.n_in)*0.5)-0.25
        self.b = np.zeros((self.n, self.m))

        self.activation = activation

    def forward_pass(self, X) -> np.ndarray:
        Z = np.dot(self.weights, X) + self.b
        Y = self.activation.run(Z)
        return Y
    
    def backwards_pass(self, learning_rate):
        d_w = 0
        self.weights -= learning_rate * d_w

class NeuralNetwork:
    def __init__(self, learning_rate, cost_function) -> None:
        self.layers = []
        self.input_shape = None
        self.output_shape = None
        
        self.learning_rate = learning_rate
        self.cost = cost_function
    
    def add_layer(self, layer):
        if self.layers == []:
            # first layer
            self.input_shape = (layer.n_in, layer.m)
            self.output_shape = (layer.n, layer.m)
            self.layers.append(layer)
        else:
            # if layer compatible
            if self.output_shape == (layer.n_in, layer.m):
                self.output_shape = (layer.n, layer.m)
                self.layers.append(layer)
            else:
                raise Exception('Layer does not match previous layer')

    def train_step(self, X):
        X = self.predict(X)
        
        for layer in self.layers[::-1]:
            X = layer.backwards_pass(self.learning_rate)
    
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X
    
if __name__ == '__main__':
    print((np.random.rand(30, 40)*0.5)-0.25)
    
