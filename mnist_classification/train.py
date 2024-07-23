from pathlib import Path
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
import os
import mlflow

mlflow.set_tracking_uri(os.environ.get("TRACKING_ARN"))
mlflow.set_experiment("MNIST Experiment")


class Net(nn.Module):
    """
    A simple convolutional neural network for MNIST classification.
    """

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


def check_mnist_files(root_dir):
    expected_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]

    mnist_dir = os.path.join(root_dir, "MNIST", "raw")

    print(f"Checking for MNIST files in: {mnist_dir}")

    if not os.path.exists(mnist_dir):
        print(f"Directory does not exist: {mnist_dir}")
        return False

    all_files_present = True
    for file in expected_files:
        file_path = os.path.join(mnist_dir, file)
        if os.path.exists(file_path):
            print(f"Found: {file}")
        else:
            print(f"Missing: {file}")
            all_files_present = False

    return all_files_present


def train_test_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Create and return train and test dataloaders for MNIST dataset. Note this
    assume the script is run on a sagemaker instance with the necesary
    dependencies installed

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test dataloaders
    """

    # Use sagemaker env var to find our data in
    sagemaker_data_root_path = os.environ.get("SM_CHANNEL_TRAINING")
    data_path = Path(sagemaker_data_root_path)

    print(f"MNIST found: {check_mnist_files(data_path)}")

    try:

        training_data = torchvision.datasets.MNIST(
            root=data_path, transform=ToTensor(), train=True, download=False
        )
        testing_data = torchvision.datasets.MNIST(
            root=data_path, transform=ToTensor(), train=False, download=False
        )

        train_dataloader = DataLoader(
            training_data, batch_size=100, shuffle=True, num_workers=1
        )
        test_dataloader = DataLoader(
            testing_data, batch_size=64, shuffle=True, num_workers=1
        )

        return train_dataloader, test_dataloader
    except Exception as e:
        raise RuntimeError(f"Error creating dataloaders: {str(e)}")


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model
        train_dataloader (DataLoader): DataLoader for training data
        loss_fn (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on (CPU or GPU)
        epoch (int): Current epoch number
    """
    model.train()

    for batch_idx, (data, target) in enumerate(train_dataloader):
        try:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print(
                    f"Train epoch: {epoch} [{batch_idx * len(data)} / {len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\t{loss.item():.6f}"
                )
                mlflow.log_metric(
                    "train_loss",
                    loss.item(),
                    step=epoch * len(train_dataloader) + batch_idx,
                )
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {str(e)}")


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    epoch: int,
) -> None:
    """
    Test the model on the test dataset.

    Args:
        model (nn.Module): The neural network model
        test_dataloader (DataLoader): DataLoader for test data
        device (torch.device): Device to test on (CPU or GPU)
        loss_fn (nn.Module): Loss function
    """
    model.eval()

    test_loss = 0
    correct = 0

    try:
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(test_dataloader.dataset)} ({100 * correct / len(test_dataloader.dataset):.0f}%\n"
        )

        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric(
            "test_accuracy", 100.0 * correct / len(test_dataloader.dataset), step=epoch
        )
    except Exception as e:
        print(f"Error in testing: {str(e)}")


def main() -> None:
    """
    Main function to run the training and testing process.
    """
    try:

        # Also available
        # mlflow.pytorch.autolog()

        with mlflow.start_run():
            train_dataloader, test_dataloader = train_test_dataloaders()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Hyperparams
            lr = 0.001
            batch_size = 100
            epochs = 10

            # MLFLOW tracking
            mlflow.log_param("device", device)
            mlflow.log_param("model_name", "MNIST CNN")
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)

            model = Net().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                train(model, train_dataloader, loss_fn, optimizer, device, epoch)
                test(model, test_dataloader, device, loss_fn, epoch)

            # log model
            mlflow.pytorch.log_model(model, "mnist_cnn")
            mlflow.set_tag(
                "Training Info",
                """Convolutional neural network in
                           PyTorch""",
            )
            mlflow.set_tag("Dataset used", "MNIST")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
