import argparse
import os
from pathlib import Path
from typing import Tuple

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor

from sagemaker_training import environment


# Import SMDataParallel PyTorch Modules, if applicable
backend = "nccl"
training_env = environment.Environment()
smdataparallel_enabled = training_env.additional_framework_parameters.get(
    "sagemaker_distributed_dataparallel_enabled", False
)
if smdataparallel_enabled:
    try:
        import smdistributed.dataparallel.torch.torch_smddp

        backend = "smddp"
        print("Using smddp as backend")
    except ImportError:
        print("smdistributed module not available, falling back to NCCL collectives.")

mlflow.set_tracking_uri(os.environ.get("TRACKING_ARN"))
mlflow.set_experiment("MNIST Experiment")

env = environment.Environment()


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


def check_mnist_files(root_dir: Path) -> bool:
    """
    Check for the presence of MNIST dataset files in the specified directory.

    This function verifies if all the required MNIST dataset files are present
    in the expected directory structure. It checks for the following files:
    - train-images-idx3-ubyte
    - train-labels-idx1-ubyte
    - t10k-images-idx3-ubyte
    - t10k-labels-idx1-ubyte

    Args:
        root_dir (Path): The root directory where the MNIST dataset is expected to be located.

    Returns:
        bool: True if all expected files are present, False otherwise.

    Prints:
        Status messages indicating which files were found or are missing.
    """
    expected_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]

    mnist_dir = root_dir / "MNIST" / "raw"

    print(f"Checking for MNIST files in: {mnist_dir}")

    if not mnist_dir.exists():
        print(f"Directory does not exist: {mnist_dir}")
        return False

    all_files_present = True
    for file in expected_files:
        file_path = mnist_dir / file
        if file_path.exists():
            print(f"Found: {file}")
        else:
            print(f"Missing: {file}")
            all_files_present = False

    return all_files_present


def train_test_datasets() -> (
    Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]
):
    """
    Create and return train and test datasets for MNIST. Will look within the
    /opt/ml/input/ path, as obtained by the SM_CHANNEL_TRAINING environmental
    variable

    Returns:
        Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]: Train
        and test datasets
    """

    # Use sagemaker env var to find our data in
    sagemaker_data_root_path = os.environ.get("SM_CHANNEL_TRAINING")
    data_path = Path(sagemaker_data_root_path)

    print(f"MNIST found: {check_mnist_files(data_path)}")
    try:
        training_dataset = torchvision.datasets.MNIST(
            root=data_path, transform=ToTensor(), train=True, download=False
        )
        testing_dataset = torchvision.datasets.MNIST(
            root=data_path, transform=ToTensor(), train=False, download=False
        )
        return training_dataset, testing_dataset
    except Exception as e:
        raise RuntimeError(f"Error creating datasets: {str(e)}")


def train_test_dataloaders(
    training_dataset: torchvision.datasets.MNIST,
    testing_dataset: torchvision.datasets.MNIST,
    train_sampler,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create and return train and test dataloaders for MNIST dataset. Note this
    assume the script is run on a sagemaker instance with the necesary
    dependencies installed

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test dataloaders
    """
    try:
        train_dataloader = DataLoader(
            training_dataset,
            batch_size=100,
            shuffle=False,  
            num_workers=0,
            sampler=train_sampler,
        )
        test_dataloader = DataLoader(
            testing_dataset, batch_size=64, shuffle=False, num_workers=1
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
    mlflow_run=None,
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
                if mlflow_run:
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
    mlflow_run=None,
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
        if mlflow_run:
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric(
                "test_accuracy", 100.0 * correct / len(test_dataloader.dataset), step=epoch
            )
    except Exception as e:
        print(f"Error in testing: {str(e)}")


def main(mlflow_run=None):
    """
    Main function to run the training and testing process.
    """
    try:
        # Training settings
        parser = argparse.ArgumentParser(description="MNIST Example")
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="For displaying smdistributed.dataparallel-specific logs",
        )

        args = parser.parse_args()

        # Configure DDP args
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_rank()

        # Hyperparams
        lr = 0.001
        batch_size = 100
        epochs = 10

        # Scale batch size by world size
        batch_size //= dist.get_world_size() // 8
        batch_size = max(batch_size, 1)

        # If we're the main process and have an MLflow run, log parameters
        if rank == 0 and mlflow_run:
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("world_size", world_size)

        # Prep dataset and data loader
        training_dataset, testing_dataset = train_test_datasets()

        train_sampler = DistributedSampler(
            training_dataset, num_replicas=world_size, rank=rank
        )

        train_dataloader, test_dataloader = train_test_dataloaders(
            training_dataset, testing_dataset, train_sampler
        )

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)
        model = DDP(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        print(f"Training on {device}")

        # Pin each GPU to a single distributed data parallel library process.
        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)

        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            train(model, train_dataloader, loss_fn, optimizer, device, epoch,
                  mlflow_run)
            if rank == 0:
                test(model, test_dataloader, device, loss_fn, epoch, mlflow_run)

        # Save checkpoint only on leader node
        if rank == 0:
            model_dir = env.model_dir
            torch.save(model.state_dict(), model_dir)

            # MLFLOW tracking (only if we have an MLflow run)
            if mlflow_run:
                mlflow.log_param("device", device)
                mlflow.log_param("model_name", "MNIST CNN")
                
                # log model
                mlflow.pytorch.log_model(model, "mnist_cnn")
                mlflow.set_tag(
                    "Training Info",
                    "Convolutional neural network in PyTorch",
                )
                mlflow.set_tag("Dataset used", "MNIST")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    
    if rank == 0:
        with mlflow.start_run() as run:
            main(run)
    else:
        main()
