from composer import Trainer
from composer.models import HuggingFaceModel

import mlflow
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

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
    model_name = "distilbert-base-uncased"
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HuggingFaceModel(model=hf_model, tokenizer=tokenizer)

    mlflow.log_param("device", device)
    mlflow.log_param("model_name", model_name)

    train_loader, val_loader, test_loader = load_and_preprocess_data(tokenizer)

    sarcasm_probability = inference(
        model=model,
        tokenizer=tokenizer,
        text=message,
    )
    print(f"Sarcasm probability: {sarcasm_probability:.2f}%")

    mlflow.set_tag(
        "Training Info",
        """Fine tuning job on distilbert for news
        headline classification""",
    )
    mlflow.log_metric("initial_accuracy", sarcasm_probability)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        max_duration="1ep",
    )

    trainer.fit()

    sarcasm_probability = inference(
        model=model,
        tokenizer=tokenizer,
        text=message,
    )
    print(f"Sarcasm probability after training: {sarcasm_probability:.2f}%")

    mlflow.log_metric("final_accuracy", sarcasm_probability)
