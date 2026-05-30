from layer import Layer
import array
import numpy as np
from cost import Cost

class NN:
    def __init__(self, input_layer: Layer, hidden_layer: array[Layer], output_layer: Layer, cost_type: str):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        pass
    
    def calculate_loss(self, cost: Cost):
        cost_func = getattr(cost, self.cost_type, None)
        if cost_func and callable(cost_func):
            return cost_func()
            

    def forward_propagation(self):
        return np.mean(self.input_layer.forward_layer())