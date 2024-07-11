# MosaicML imports
from composer import Trainer
from composer.models import HuggingFaceModel
from composer.loggers import MLFlowLogger

import mlflow
import torch

# Hugging face imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

# Personal imports
from inference import inference
from preprocess import load_and_preprocess_data

mlflow.set_tracking_uri(os.environ.get("TRACKING_ARN"))
mlflow.set_experiment("Presentation")

message = """Breaking news! Leading world class scientists at MIT have
gathered to discuss whether water is, in fact, wet."""

print("The following headline will be used to test our model")
print(message)

# Use cuda if available
if torch.cuda.is_available():
    device = "gpu"
else:
    device = "cpu"
print(f"Using device: {device}")


with mlflow.start_run():
    # Define our hf model and use composer to wrap it
    model_name = "distilbert-base-uncased"
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HuggingFaceModel(model=hf_model, tokenizer=tokenizer)

    mlflow.log_param("device", device)
    mlflow.log_param("model_name", model_name)
    # Load and preprocess data
    train_loader, val_loader, test_loader = load_and_preprocess_data(tokenizer)

    # Before we train our model, let's test how good it is at the task
    sarcasm_probability = inference(
        model=model,
        tokenizer=tokenizer,
        text=message,
    )
    print(f"Sarcasm probability: {sarcasm_probability:.2f}%")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag(
        "Training Info",
        """Fine tuning job on distilbert for news
        headline classification""",
    )
    mlflow.log_metric("initial_accuracy", sarcasm_probability)
    mlflow_logger = MLFlowLogger()

    # Set up the trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        max_duration="1ep",
        loggers=[mlflow_logger]
    )

    # Train the model
    trainer.fit()

    # After training, test again
    sarcasm_probability = inference(
        model=model,
        tokenizer=tokenizer,
        text=message,
    )
    print(f"Sarcasm probability after training: {sarcasm_probability:.2f}%")

    mlflow.log_metric("final_accuracy", sarcasm_probability)
