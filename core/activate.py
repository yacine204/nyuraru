from layer import Layer
import numpy as mp
import array

class Activate:
    def ReLU(layer: Layer) -> array[float]:
        return np.max(0, layer.forward_layer_())
    
    def Sigmoid(layer: Layer) -> array[float]:
        return 1/(1+np.exp(-layer.forward_layer_()))
    
    def Softmax(layer: Layer) -> array[float]:
        x = layer.forward_layer() - np.max(layer.forward_layer())
        exp_x = np.exp(layer.forward_layer())
        return x/exp_x
