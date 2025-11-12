# src/data_utils.py
from datasets import load_dataset

def load_csv(train_path: str, test_path: str):
    train = load_dataset("csv", data_files=train_path)["train"]
    test  = load_dataset("csv", data_files=test_path)["train"]
    return train, test

def build_messages(instruction: str, context: str | None = None):
    content = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return [{"role": "user", "content": content}]
