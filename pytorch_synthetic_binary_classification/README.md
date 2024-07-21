# Binary Classification with Synthetic Data

## Overview
This training job trains a simple binary classification model using synthetic data. The model is a single-layer neural network that takes 10 input features and outputs a binary classification.

## Dataset
- Source: Synthetically generated within the script
- Format: NumPy arrays converted to PyTorch tensors
- Size: 2000 training samples, 400 validation samples

## Environment
- Framework: PyTorch
- SageMaker Instance Type: Not specified, but supports both CPU and GPU (CUDA) instances

## Training
- Entry Point: train.py
- Hyperparameters:
  - batch_size: 32
  - epochs: 5
  - learning_rate: 0.1
  - optimizer: SGD

Additional Notes:
- The script supports both local training and distributed training using SageMaker's distributed training library.
- The model architecture is a simple fully connected layer with sigmoid activation.
- The script uses Binary Cross Entropy Loss for training.
- The trained model is saved as 'model.pth' in the SageMaker model directory.
