# ニューラル (under development)

Deep-learning library for visualing neural networks training results and tests depending on cost/loss functions chosen

## What it does

### Current
- create your custom neural network
- choose activation and Loss functions
- train it on a npy dataset
- test it to evaluate it

### Planned
- visualize training
- real-time tests
- performance benchmarks between different parameters

## How it works:

```py
from core.nn import NN
from core.layer import Layer
import numpy as np

# dataset preparation
x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')

x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

y_onehot = np.zeros((60000, 10))
y_onehot[np.arange(60000), y_train] = 1
y_test_onehot = np.zeros((10000, 10))
y_test_onehot[np.arange(10000), y_test] = 1

# nn preparation
input_layer = Layer(28 * 28)
hidden_layer = [
	Layer(128, activation="relu"),
	Layer(64, activation="relu"),
]
output_layer = Layer(10, activation="softmax")
neural_network = NN(input_layer, hidden_layer, output_layer, "CCE")

neural_network.train(40, 0.01, x_train, y_onehot, batch_size=128, shuffle=True)

test_loss, test_accuracy = neural_network.evaluate(x_test, y_test_onehot)
print(f"Test Cost: {test_loss:.4f}, Test Accuracy: {test_accuracy:.3f}")
```
