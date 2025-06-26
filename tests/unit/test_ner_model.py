import os
import tempfile
from transformers import AutoTokenizer

from src.models.ner_model import load_ner_model, save_ner_model, get_ner_pipeline

def test_load_ner_model():
    model_checkpoint = "xlm-roberta-base"
    num_labels = 3
    id2label = {0: "O", 1: "B-LOC", 2: "I-LOC"}
    label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    model = load_ner_model(model_checkpoint, num_labels, id2label, label2id)
    assert model.config.num_labels == num_labels
    assert model.config.id2label == id2label
    assert model.config.label2id == label2id

def test_save_and_load_ner_model(tmp_path):
    model_checkpoint = "xlm-roberta-base"
    num_labels = 3
    id2label = {0: "O", 1: "B-LOC", 2: "I-LOC"}
    label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    model = load_ner_model(model_checkpoint, num_labels, id2label, label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    save_dir = tmp_path / "ner_model"
    save_ner_model(model, tokenizer, str(save_dir))
    # Check files exist
    assert (save_dir / "config.json").exists()
    assert (save_dir / "tokenizer.json").exists() or (save_dir / "tokenizer_config.json").exists()

def test_get_ner_pipeline():
    model_checkpoint = "xlm-roberta-base"
    num_labels = 3
    id2label = {0: "O", 1: "B-LOC", 2: "I-LOC"}
    label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2}
    model = load_ner_model(model_checkpoint, num_labels, id2label, label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    nlp = get_ner_pipeline(model, tokenizer)
    result = nlp("Addis Ababa is the capital of Ethiopia.")
    assert isinstance(result, list)
    assert all("entity_group" in ent for ent in result)