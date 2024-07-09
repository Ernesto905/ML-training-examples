# MosaicML imports
from composer import Trainer
from composer.models import HuggingFaceModel

import torch

# Hugging face imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Personal imports
from inference import inference
from preprocess import load_and_preprocess_data

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

# Define our hf model and use composer to wrap it
model_name = "distilbert-base-uncased"
hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HuggingFaceModel(model=hf_model, tokenizer=tokenizer)

# Load and preprocess data
train_loader, val_loader, test_loader = load_and_preprocess_data(tokenizer)

# Before we train our model, let's test how good it is at the task
sarcasm_probability = inference(
    model=model,
    tokenizer=tokenizer,
    text=message,
)
print(f"Sarcasm probability: {sarcasm_probability:.2f}%")

# Set up the trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    max_duration="2ep",
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
