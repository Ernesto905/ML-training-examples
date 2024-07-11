import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Conditionally import SageMaker's distributed training library
try:
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import (
        DistributedDataParallel as DDP,
    )

    SAGEMAKER_DISTRIBUTED = True
except ImportError:
    SAGEMAKER_DISTRIBUTED = False
    print("SageMaker Distributed Training library not found. Running in local mode.")


def setup_distributed():
    if SAGEMAKER_DISTRIBUTED:
        dist.init_process_group()
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0  # world_size = 1, rank = 0 for local mode


# Define a simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


def generate_synthetic_data(size=100):
    X = np.random.normal(size=(size, 10)).astype(np.float32)

    # Create a linear pattern: y = Xw + b, where `w` is a weight vector and b is bias
    weights = np.array([2.0] + [0] * (10 - 1), dtype=np.float32)
    bias = np.array([0.5], dtype=np.float32)
    y = X @ weights + bias

    # Convert the pattern to labels between 0 and 1.
    y = (y > y.mean()).astype(np.float32)
    y = y.reshape(-1, 1)  # Ensure target is of shape (size, 1)

    return X, y


def train(device, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(loader)


def evaluate(device, loader, model, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return total_loss / len(loader), correct / total


def main():
    world_size, rank = setup_distributed()

    # Setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}: Training on device: {device}")

    # Generating synthetic training and validation data
    X_train, y_train = generate_synthetic_data(2000)
    X_val, y_val = generate_synthetic_data(400)

    # Create DataLoader instances
    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model, criterion, and optimizer
    model = Net().to(device)
    if SAGEMAKER_DISTRIBUTED:
        model = DDP(
            model
        )  # Wrap the model with DistributedDataParallel only in distributed mode
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 5
    for epoch in range(epochs):
        train_loss = train(device, train_loader, model, criterion, optimizer)
        val_loss, val_accuracy = evaluate(device, val_loader, model, criterion)
        if rank == 0:  # Only print from the main process
            print(
                f"Epoch: {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

    # Save the trained model
    if rank == 0:  # Only save from the main process
        model_dir = os.environ.get("SM_MODEL_DIR", ".")
        model_path = os.path.join(model_dir, "model.pth")
        if SAGEMAKER_DISTRIBUTED:
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
