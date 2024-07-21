# Sarcasm Detection Training Job

## Overview
This training job fine-tunes a DistilBERT model for sarcasm detection in news headlines. The model is trained to classify headlines as either sarcastic or not sarcastic.

## Dataset
- Source: Sarcasm_Headlines_Dataset.json (loaded from SageMaker input channel)
- Format: JSON
- Size: 1500 examples (limited from the original dataset)
- [Kaggle link](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection?resource=download)

## Environment
- Framework: PyTorch with Hugging Face Transformers and MosaicML Composer
- SageMaker Instance Type: Not specified, but uses GPU if available

## Training
- Entry Point: train.py
- Hyperparameters:
  - model_name: distilbert-base-uncased
  - max_length: 128 (for tokenization)
  - batch_size: 32
  - max_duration: 1 epoch
  - num_labels: 2 (sarcastic or not sarcastic)

Additional Notes:
- Uses MLflow for experiment tracking and logging
- Includes pre-training and post-training inference to measure improvement
- Utilizes custom SarcasmDataset class for data handling
- Implements train/validation/test split (80% / 8% / 12%)
