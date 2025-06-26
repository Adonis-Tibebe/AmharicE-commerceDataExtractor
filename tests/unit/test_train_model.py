import pytest
from datasets import DatasetDict, Dataset

from src.core.train import train_model

def get_dummy_dataset():
    # Minimal dummy dataset for NER
    data = {
        "tokens": [["Addis", "Ababa"], ["Ethiopia"]],
        "ner_tags": [[1, 2], [0]],
    }
    features = {
        "tokens": data["tokens"],
        "ner_tags": data["ner_tags"],
    }
    train_ds = Dataset.from_dict(features)
    val_ds = Dataset.from_dict(features)
    return DatasetDict({"train": train_ds, "validation": val_ds})

class DummyMetric:
    def compute(self, predictions, references):
        return {
            "overall_precision": 1.0,
            "overall_recall": 1.0,
            "overall_f1": 1.0,
            "overall_accuracy": 1.0,
        }

def test_train_model_runs(monkeypatch):
    # Use a tiny model for fast testing
    model_checkpoint = "sshleifer/tiny-distilroberta-base"
    label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    id2label = {0: "O", 1: "B-LOC", 2: "I-LOC"}
    unique_labels = ["O", "B-LOC", "I-LOC"]
    metric = DummyMetric()
    raw_dataset = get_dummy_dataset()

    # Patch TrainingArguments to reduce epochs and steps for speed
    from transformers import TrainingArguments
    monkeypatch.setattr(
        "src.core.train.TrainingArguments",
        lambda *args, **kwargs: TrainingArguments(
            output_dir="./tmp",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            save_steps=1,
            evaluation_strategy="no",
            save_strategy="no",
            report_to=[],
        )
    )

    metrics, model, tokenizer = train_model(
        model_checkpoint,
        raw_dataset,
        label2id,
        id2label,
        unique_labels,
        metric,
        return_model=True
    )
    assert "overall_f1" in metrics or "f1" in metrics
    assert model is not None
    assert tokenizer is not None