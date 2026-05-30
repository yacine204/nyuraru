import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def main() -> None:
    x_train = np.load("./dataset/x_train.npy")
    y_train = np.load("./dataset/y_train.npy")
    x_test = np.load("./dataset/x_test.npy")
    y_test = np.load("./dataset/y_test.npy")

    x_train = x_train.reshape(60000, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(10000, 784).astype(np.float32) / 255.0

    y_train_onehot = to_one_hot(y_train, 10)
    y_test_onehot = to_one_hot(y_test, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train_onehot)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(y_test_onehot)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def cce_with_onehot(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

    start_time = time.perf_counter()
    model.train()
    for _ in range(10):
        for inputs, targets_onehot in train_loader:
            inputs = inputs.to(device)
            targets = targets_onehot.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = cce_with_onehot(logits, targets)
            loss.backward()
            optimizer.step()

    train_duration = time.perf_counter() - start_time

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, targets_onehot in test_loader:
            inputs = inputs.to(device)
            targets = targets_onehot.to(device)

            logits = model(inputs)
            loss = cce_with_onehot(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == targets.argmax(dim=1)).sum().item()
            total_count += inputs.size(0)

    test_loss = total_loss / total_count
    test_accuracy = total_correct / total_count

    print(f"Train Time: {train_duration:.2f}s")
    print(f"Test Cost: {test_loss:.4f}, Test Accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()
