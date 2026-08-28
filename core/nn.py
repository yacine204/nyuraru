import time

from .layer import Layer
from .activate import Activate
import numpy as np
import os
from .cost import Cost

class NN:
    def __init__(self, input_layer: Layer, hidden_layer: list[Layer], output_layer: Layer, cost_function: str, model_file_path: str=None):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.layers = [self.input_layer] + self.hidden_layer + [self.output_layer]
        self.cost_function = cost_function.upper()
        self.huber_delta = 1.0
        self.cost = Cost(self.cost_function, threshold=self.huber_delta)
        self.model_file_path = model_file_path

        self.layer_sizes = [self._layer_size(layer) for layer in self.layers]
        self.activations = self._build_activations()

        self.weights, self.biases = self._init_params()

        if self.model_file_path is not None:
            model = self._load_model(self.model_file_path)
            if model is not None:
                self.weights, self.biases = model

        self._sync_layer_params()
        self.cache_a = []
        self.cache_z = []

    def _layer_size(self, layer: Layer) -> int:
        if hasattr(layer, "n_nodes") and layer.n_nodes is not None:
            if isinstance(layer.n_nodes, int):
                return layer.n_nodes
            return len(layer.n_nodes)
        if hasattr(layer, "nodes") and layer.nodes is not None:
            return len(layer.nodes)
        raise ValueError("Layer size could not be determined")

    def _build_activations(self) -> list[str]:
        activations = []
        for idx, layer in enumerate(self.layers[1:], start=1):
            activation = getattr(layer, "activation", None)
            if activation:
                activations.append(str(activation).lower())
                continue
            is_output = idx == len(self.layers) - 1
            if is_output:
                if self.cost_function == "BCE":
                    activations.append("sigmoid")
                elif self.cost_function == "CCE":
                    activations.append("softmax")
                else:
                    activations.append("linear")
            else:
                activations.append("relu")
        return activations

    def _init_params(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights = []
        biases = []
        for i in range(1, len(self.layer_sizes)):
            n_in = self.layer_sizes[i - 1]
            n_out = self.layer_sizes[i]
            weight = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            bias = np.zeros(n_out)
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def _sync_layer_params(self) -> None:
        for idx, layer in enumerate(self.layers[1:]):
            layer.weight = self.weights[idx]
            layer.bias = self.biases[idx]

    def _activation(self, x: np.ndarray, kind: str) -> np.ndarray:
        if kind == "relu":
            return Activate.ReLU(x)
        if kind == "sigmoid":
            return Activate.Sigmoid(x)
        if kind == "softmax":
            return Activate.Softmax(x)
        return x

    def _activation_derivative(self, a: np.ndarray, kind: str, z: np.ndarray | None = None) -> np.ndarray:
        if kind == "relu":
            return (z > 0).astype(float)
        if kind == "sigmoid":
            return a * (1 - a)
        if kind == "softmax":
            return a * (1 - a)
        return np.ones_like(a)

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.cost.execute(y_true, y_pred)

    def _forward_propagation(self, x_batch: np.ndarray) -> np.ndarray:
        self.cache_a = [x_batch]
        self.cache_z = []
        self.input_layer.nodes = x_batch

        a = x_batch
        for idx, (weight, bias, activation) in enumerate(zip(self.weights, self.biases, self.activations)):
            z = np.dot(a, weight) + bias
            self.cache_z.append(z)
            a = self._activation(z, activation)
            self.cache_a.append(a)
            self.layers[idx + 1].nodes = a
        return a

    def _backward_propagation(
        self,
        y_true: np.ndarray,
        learning_rate: float,
        clip_value: float | None = 5.0,
    ) -> None:
        m = y_true.shape[0]
        y_pred = self.cache_a[-1]
        output_activation = self.activations[-1]

        if self.cost_function in {"BCE", "CCE"} and output_activation in {"sigmoid", "softmax"}:
            delta = y_pred - y_true
        elif self.cost_function == "MAE":
            delta = np.sign(y_pred - y_true) * self._activation_derivative(y_pred, output_activation, self.cache_z[-1])
        else:
            delta = (y_pred - y_true) * self._activation_derivative(y_pred, output_activation, self.cache_z[-1])

        for idx in reversed(range(len(self.weights))):
            a_prev = self.cache_a[idx]
            dW = np.dot(a_prev.T, delta) / m
            db = np.mean(delta, axis=0)

            if clip_value is not None:
                dW = np.clip(dW, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)

            self.weights[idx] -= learning_rate * dW
            self.biases[idx] -= learning_rate * db

            if idx > 0:
                activation = self.activations[idx - 1]
                delta = np.dot(delta, self.weights[idx].T) * self._activation_derivative(
                    self.cache_a[idx], activation, self.cache_z[idx - 1]
                )

                if clip_value is not None:
                    delta = np.clip(delta, -clip_value, clip_value)

        self._sync_layer_params()

    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            return float(np.mean((y_pred_flat >= 0.5) == y_true_flat))
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return float(np.mean(y_true_labels == y_pred_labels))

    def _save_model(self, file_path: str = None) -> None:
        if file_path is None:
            print(f"file_path not provided, overiding {file_path}")
            file_path = self.model_file_path

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(
            file_path,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
            layer_sizes=np.array(self.layer_sizes),
            activations=np.array(self.activations),
            cost_function=self.cost_function,
        )
        print(f"Model saved to {file_path}")

    def _load_model(self, file_path: str) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        if not os.path.exists(file_path):
            print(f"{file_path} doesn't seem like a legit path")
            return None

        data = np.load(file_path, allow_pickle=True)
        if "weights" in data and "biases" in data:
            weights = data["weights"].tolist()
            biases = data["biases"].tolist()
            if len(weights) == len(self.layer_sizes) - 1:
                print(f"{file_path} loaded")
                return weights, biases
        return None

    def evaluate(self, x_data, y_data) -> tuple[float, float]:
        x_eval = np.asarray(x_data)
        y_eval = np.asarray(y_data)
        predictions = self._forward_propagation(x_eval)
        loss = self._calculate_loss(y_eval, predictions)
        accuracy = self._compute_accuracy(y_eval, predictions)
        return loss, accuracy

    def train(
        self,
        epochs: int,
        learning_rate: float,
        training_batch,
        y_batch,
        batch_size: int = 128,
        shuffle: bool = True,
        visualize: bool = False,
        file_path: str = None,
    ) -> None:

        if file_path is None:
            return ValueError("include file_path to specify where to save model")

        x_data = np.asarray(training_batch)
        y_data = np.asarray(y_batch)
        n_samples = x_data.shape[0]

        viz = None
        if visualize:
            from .visualizer import Visualizer
            viz = Visualizer(self.layer_sizes, self.activations[-1])
            input_size = self.layer_sizes[0]
            side = int(round(input_size ** 0.5))
            if side * side == input_size:
                    viz.set_input_grid(side, side)

        stop_early = False
        for epoch in range(epochs):
            if stop_early:
                break

            if shuffle:
                indices = np.random.permutation(n_samples)
                x_data = x_data[indices]
                y_data = y_data[indices]

            batch_losses = []
            batch_accuracies = []
            for start in range(0, n_samples, batch_size):
                if viz and viz.should_close():
                    stop_early = True
                    break

                end = start + batch_size
                x_batch = x_data[start:end]
                y_batch_slice = y_data[start:end]

                predictions = self._forward_propagation(x_batch)
                loss = self._calculate_loss(y_batch_slice, predictions)
                self._backward_propagation(y_batch_slice, learning_rate)
                accuracy = self._compute_accuracy(y_batch_slice, predictions)
                if viz:
                    viz.push_activations(self.cache_a)
                    viz.frame()

                batch_losses.append(loss)
                batch_accuracies.append(accuracy)

            epoch_loss = float(np.mean(batch_losses))
            epoch_accuracy = float(np.mean(batch_accuracies))
            print(f"Epoch: {epoch + 1}/{epochs}, Cost: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.3f}")
        if viz:
            viz.close()

        self._save_model(file_path)
        print("Training complete")

    def predict_loop(self, viz=None, grid_w: int = 28, grid_h: int = 28, auto_predict: bool = True) -> None:
        own_viz = viz is None
        if own_viz:
            from .visualizer import Visualizer
            viz = Visualizer(self.layer_sizes, self.activations[-1])
            side = int(round(self.layer_sizes[0] ** 0.5))
            if side * side == self.layer_sizes[0]:
                viz.set_input_grid(side, side)
        else:
            viz.reset_display() 
        print("[predict_loop] entering loop")   
        frame_count = 0                          
        last_probs = None
        last_probs = None
        last_prediction_time = 0
        prediction_cooldown = 0.1

        while not viz.should_close():
            frame_count += 1

            if viz.clear_pressed():
                viz.clear_board()
                last_probs = None
                if hasattr(viz, 'reset_board_modified'):
                    viz.reset_board_modified()

            if auto_predict and viz.board_modified():
                current_time = time.time()
                if current_time - last_prediction_time > prediction_cooldown:
                    print("Board modified - predicting...")
                    pixels = viz.get_board_pixels(grid_w, grid_h)
                    
                 
                    if np.sum(pixels) > 0.5:  
                        x = pixels.reshape(1, -1)
                        probs = self._forward_propagation(x)[0]
                        last_probs = probs
                        viz.push_activations([x] + self.cache_a[1:])
                        
                        digit = int(np.argmax(probs))
                        confidence = float(probs[digit])
                        print(f"Auto-predicted: {digit} ({confidence * 100:.1f}%)")
                        last_prediction_time = current_time
                        viz.reset_board_modified()

            if viz.predict_pressed():
                pixels = viz.get_board_pixels(grid_w, grid_h)  
                x = pixels.reshape(1, -1)

                probs = self._forward_propagation(x)[0]
                last_probs = probs

                
                viz.push_activations([x] + self.cache_a[1:])

                digit = int(np.argmax(probs))
                confidence = float(probs[digit])
                print(f"Predicted: {digit} ({confidence * 100:.1f}%)")

            pred = None
            if last_probs is not None:
                digit = int(np.argmax(last_probs))
                confidence = float(last_probs[digit])
                pred = (digit, confidence)

            viz.frame(prediction=pred)
        print(f"[predict_loop] exited after {frame_count} frames")
        if own_viz:
            viz.close()