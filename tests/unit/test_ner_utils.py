import os
import tempfile
import pytest
from transformers import AutoTokenizer

from src.ner_utils import load_conll, tokenize_and_align_labels, compute_metrics

def test_load_conll():
    # Create a temporary CoNLL file
    content = "አዲስ O O B-LOC\nአበባ O O I-LOC\n\nሰላም O O O\n"
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name
    sentences, labels = load_conll(temp_path)
    os.remove(temp_path)
    assert sentences == [["አዲስ", "አበባ"], ["ሰላም"]]
    assert labels == [["B-LOC", "I-LOC"], ["O"]]

def test_tokenize_and_align_labels():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    examples = {
        "tokens": [["Addis", "Ababa"]],
        "ner_tags": [["B-LOC", "I-LOC"]]
    }
    label2id = {"B-LOC": 0, "I-LOC": 1, "O": 2}
    tokenized = tokenize_and_align_labels(examples, tokenizer, label2id)
    # Check that labels are aligned and special tokens are -100
    assert "labels" in tokenized
    assert any(-100 in seq for seq in tokenized["labels"])

def test_compute_metrics():
    # Simulate logits and labels for 2 samples, 3 classes
    import numpy as np
    logits = np.array([[[2, 1, 0], [0, 2, 1]], [[1, 2, 0], [2, 0, 1]]])
    labels = np.array([[0, 1], [1, -100]])
    unique_labels = ["A", "B", "O"]
    class DummyMetric:
        def compute(self, predictions, references):
            return {
                "overall_precision": 1.0,
                "overall_recall": 1.0,
                "overall_f1": 1.0,
                "overall_accuracy": 1.0,
            }
    metric = DummyMetric()
    result = compute_metrics((logits, labels), unique_labels, metric)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0