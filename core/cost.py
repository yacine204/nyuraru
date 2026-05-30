from layer import Layer
import numpy as np
import array

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
    
