import numpy as np
from typing import Callable

class Layer:
    def __init__(self, input_size: tuple[int, int], activation: Callable[[np.ndarray], np.ndarray]) -> None:
        self.n_x = input_size[0]
        self.m = input_size[1]
        
        # needs initialization
        self.weights = np.zeros((1,self.n_x))
        self.b = 0

        self.activation = activation

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        Z = np.dot(self.weights.T, X) + self.b
        Y = self.activation(Z)
        return Y
    
    def backwards_pass(self):
        pass
    
    def initialize_weights(self):
        pass