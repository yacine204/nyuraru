import array
import numpy as np
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

class Activate:
    def ReLU(layer: Layer) -> array[float]:
        return np.max(0, layer.forward_layer_())
    
    def Sigmoid(layer: Layer) -> array[float]:
        return 1/(1+np.exp(-layer.forward_layer_()))
    
    def Softmax(layer: Layer) -> array[float]:
        x = layer.forward_layer() - np.max(layer.forward_layer())
        exp_x = np.exp(layer.forward_layer())
        return x/exp_x

class Cost:
    def __init__(self, layer: Layer, output_layer: Layer, threshhold: float, n_classes: int):
        self.layer.nodes = layer
        self.output_layer = output_layer
        if threshhold is not None: 
            self.threshhold = threshhold
        if n_classes is not None:
            self.n_classes = n_classes
        pass

    # REGRESSION:
    
    # Mean Squared Error
    def MSE(self):
        return (self.layer.nodes - self.output_layer)**2
    
    # Mean Absolute Error
    def MAE(self):
        return 1/self.layer.nodes * np.mean((self.layer.nodes - self.output_layer))
    
    # Huber Loss
    def Huber(self):
        if (self.layer.nodes - self.output_layer) < self.threshhold:
            return 1/2* self.MSE()
        else: 
            return self.threshhold*(np.abs((self.layer.nodes-self.output_layer)) - 1/2*(self.threshhold))
    
    # CLASSIFICATION:

    # Binary Cross-Entropy
    def BCE(self):
        return 1/len(self.layer.nodes) * (np.mean((self.layer.nodes*np.log(self.output_layer)),(np.dot((1-self.layer.nodes),(np.log(1-self.output_layer))))))
    
    # Categorical Cross-Entropy
    def CCE(self):
        y_c = np.zeros(len(self.layer.nodes))
        y_c = [1 if x == self.output_layer  else 0 for x in y_c]
        return - self.n_classes * np.mean(y_c, (np.log(self.output_layer)))
    

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