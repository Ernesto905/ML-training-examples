import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["headline"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(item["is_sarcastic"], dtype=torch.long),
        }


def get_data_from_sagemaker_input():
    """
    Retrieves the JSON data from the SageMaker input directory.

    Returns:
    list: A list of dictionaries, each representing a JSON object from the file.
    """
    training_path = os.environ["SM_CHANNEL_TRAINING"]

    json_file_path = os.path.join(training_path, "Sarcasm_Headlines_Dataset.json")

    with open(json_file_path, "r") as f:
        return [json.loads(line) for line in f]


def load_and_preprocess_data(tokenizer):
    """
    Tokenizes our data and returns training, cross validation, and testing set
    as pytorch dataloader objects.
    """
    data = get_data_from_sagemaker_input()

    data = data[:1500]

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_dataset = SarcasmDataset(train_data, tokenizer, max_length=128)
    val_dataset = SarcasmDataset(val_data, tokenizer, max_length=128)
    test_dataset = SarcasmDataset(test_data, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader
