from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layer import Layer


class Activate:
    @staticmethod
    def _get_input(x_or_layer: np.ndarray | "Layer") -> np.ndarray:
        if hasattr(x_or_layer, "forward_layer"):
            return x_or_layer.forward_layer()
        return np.asarray(x_or_layer)

    @staticmethod
    def ReLU(x_or_layer: np.ndarray | "Layer") -> np.ndarray:
        x = Activate._get_input(x_or_layer)
        return np.maximum(0, x)

    @staticmethod
    def Sigmoid(x_or_layer: np.ndarray | "Layer") -> np.ndarray:
        x = Activate._get_input(x_or_layer)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def Softmax(x_or_layer: np.ndarray | "Layer") -> np.ndarray:
        x = Activate._get_input(x_or_layer)
        if x.ndim == 1:
            shifted = x - np.max(x)
            exp_x = np.exp(shifted)
            return exp_x / np.sum(exp_x)
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
