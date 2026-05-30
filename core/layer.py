import numpy as np
import array
import math

class Layer:
    def __init__(self, n_nodes: array[float], nodes: array[float], weight: array[float], bias: array[float]):
        self.n_nodes = n_nodes
        self.nodes = nodes
        self.weight = weight
        if bias is not None:
            self.bias = bias
        pass
  
    def forward_layer(self):
        if self.bias is not None:
            self.n_nodes * self.weight + self.bias
        else: 
            return None
        
    def activate(self, activation_type):
        match activation_type:
            case 'ReLU':
                return np.max(0, self.forward_layer())
            case 'Sigmoid':
                return 1/(1+math.exp(-self.forward_layer()))
            case 'Softmax':
                x = self.forward_layer() - np.max(self.forward_layer())
                exp_x = np.exp(self.forward_layer())
                return exp_x / np.sum(exp_x)
