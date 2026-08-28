import numpy as np
from core.nn import NN 
from core.layer import Layer

cat = np.load('./dataset/cat/cat.npy', allow_pickle=True)
dog = np.load('./dataset/dog/dog.npy', allow_pickle=True)

n_samples = 5000

np.random.seed(42)
cat_indices = np.random.choice(len(cat), n_samples, replace=False)
dog_indices = np.random.choice(len(dog), n_samples, replace=False)

cat_data = cat[cat_indices]
dog_data = dog[dog_indices]

cat_data = cat_data/255.0
dog_data = dog_data/255.0

x_data = np.concatenate([cat_data, dog_data], axis=0)
y_data = np.zeros((len(cat_data) + len(dog_data), 2))
# 0 : cat , 1 : dog
y_data[:len(cat_data), 0] = 1  
y_data[len(cat_data):, 1] = 1

indices = np.random.permutation(len(x_data))
x_data = x_data[indices]
y_data = y_data[indices]

n_samples = len(x_data)
n_train = int(n_samples * 0.7)
n_valid = int(n_samples * 0.15)

x_train = x_data[:n_train]
y_train = y_data[:n_train]

x_valid = x_data[n_train:n_train + n_valid]
y_valid = y_data[n_train:n_train + n_valid]

x_test = x_data[n_train + n_valid:]
y_test = y_data[n_train + n_valid:]



input_layer = Layer(784)
hidden_layer = [
    Layer(256, activation="relu"),
    Layer(128, activation="relu"),
    Layer(64, activation="relu"),
]
output_layer = Layer(2, activation="softmax")

nn = NN(input_layer, hidden_layer, output_layer, "CCE", "weights/dog_cat.npz")

# nn.train(
#     epochs=5,
#     learning_rate=0.01,
#     training_batch=x_train,
#     y_batch=y_train,
#     batch_size=128,
#     shuffle=True,
#     visualize=True,
#     file_path="weights/dog_cat.npz",
# )

loss, acc = nn.evaluate(x_data=x_valid, y_data=y_valid)
print(f"loss: {loss}, acc: {acc}")

test_loss, test_acc = nn.evaluate(x_data=x_test, y_data=y_test)
print(f"loss: {test_loss}, acc: {test_acc}")


nn.predict_loop()