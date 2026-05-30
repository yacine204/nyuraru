import numpy as np
import array
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .activate import Activate


class Layer:
    def __init__(
        self,
        n_nodes: int,
        activation: str | None = None,
        nodes: np.ndarray | None = None,
        weight: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ):
        self.n_nodes = n_nodes
        self.activation = activation
        self.nodes = nodes
        self.weight = weight
        self.bias = bias

    def forward_layer(self):
        if self.nodes is None or self.weight is None:
            return None
        if self.bias is not None:
            return np.dot(self.nodes, self.weight) + self.bias
        return np.dot(self.nodes, self.weight)

    def activate(self, activation_type):
        from .activate import Activate

        activation_function = getattr(Activate, activation_type, None)
        if activation_function and callable(activation_function):
            return activation_function(self)
        raise ValueError(f"activation function {activation_type} is not found")
