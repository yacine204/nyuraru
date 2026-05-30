from __future__ import annotations

import numpy as np
import array
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layer import Layer

class Activate:
    def ReLU(layer: Layer) -> array[float]:
        return np.max(0, layer.forward_layer_())
    
    def Sigmoid(layer: Layer) -> array[float]:
        return 1/(1+np.exp(-layer.forward_layer_()))
    
    def Softmax(layer: Layer) -> array[float]:
        x = layer.forward_layer() - np.max(layer.forward_layer())
        exp_x = np.exp(layer.forward_layer())
        return x/exp_x
