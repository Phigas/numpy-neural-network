import numpy as np

class Layer:
    def __init__(self, input_size: tuple = ()) -> None:
        self.weights = 0
    
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass