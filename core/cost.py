import numpy as np
from enum import Enum


class CostFunction(Enum):
    MSE = "MSE"
    MAE = "MAE"
    HUBER = "HUBER"
    BCE = "BCE"
    CCE = "CCE"


class Cost:
    def __init__(self, cost_function: CostFunction | str, threshold: float = 1.0, n_classes: int | None = None):
        self.cost_function_name = (
            cost_function.value if isinstance(cost_function, CostFunction) else str(cost_function).upper()
        )
        self.threshold = threshold
        self.n_classes = n_classes

    # REGRESSION:
    def _MSE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def _MAE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def _HUBER(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.threshold)
        linear = abs_diff - quadratic
        return float(np.mean(0.5 * quadratic ** 2 + self.threshold * linear))

    # CLASSIFICATION:
    def _BCE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def _CCE(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    def execute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        method = getattr(self, f"_{self.cost_function_name}", None)
        if method and callable(method):
            return method(y_true, y_pred)
        raise ValueError(f"Method '{self.cost_function_name}' not found")