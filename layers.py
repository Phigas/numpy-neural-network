import numpy as np

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
        self.Z = np.dot(self.weights, X) + self.b
        self.a = self.activation.run(self.Z) # possible to avoid keeping a saved by recalculating it
        return self.a
    
    def backwards_pass(self, learning_rate, d_a):
        d_z = d_a * self.activation.derivate(self.Z)
        
        new_d_a = d_z * self.weights
        
        d_w = np.dot(d_z, self.a.T)
        self.weights -= learning_rate * d_w
        
        d_b = d_z
        self.b -= learning_rate * d_b
        
        return new_d_a

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

    def train_step(self, X, Y):
        X = self.predict(X)
        
        loss = self.cost.run(X, Y)
        
        d_a = self.cost.derivative(X, Y)
        
        for layer in self.layers[::-1]:
            d_a = layer.backwards_pass(self.learning_rate, d_a)
            
        return loss
    
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X
    
if __name__ == '__main__':
    sigma = lambda x: 1/(1+np.exp(-x))
    sigma_d = lambda x: np.exp(-x)/(1+np.exp(-x))**2
    sigmoid = Function(sigma, sigma_d)
    
    print(sigmoid.derivate(0))
    
