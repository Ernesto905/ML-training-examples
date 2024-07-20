# MNIST Classification with CNN

## Overview
This training job trains a Convolutional Neural Network (CNN) for MNIST digit classification. The project is a work in progress, aiming to include data parallelism on SageMaker and experiment tracking with MLFlow.

## Dataset
- Source: MNIST dataset (downloaded via torchvision)
- Format: Image data (28x28 grayscale images)
- Size: 60,000 training samples, 10,000 test samples

## Environment
- Framework: PyTorch
- SageMaker Instance Type: Not specified, but supports both CPU and GPU instances

## Training
- Entry Point: train.py
- Hyperparameters:
  - batch_size: 100 (training), 64 (testing)
  - epochs: 10
  - learning_rate: 0.001
  - optimizer: Adam

Additional Notes:
- The script includes a simple CNN architecture for MNIST classification.
- Data parallelism with SageMaker is planned but not yet implemented.
- MLFlow integration for experiment tracking is planned but not yet implemented.
- A 'notebooks' directory is included for experimentation and tweaking of the CNN. Users are encouraged to explore and modify the model architecture using these notebooks.
- The script includes error handling and logging for better debugging.
- The model uses Cross Entropy Loss for training.
- The training process includes both training and testing phases, with accuracy reporting.
