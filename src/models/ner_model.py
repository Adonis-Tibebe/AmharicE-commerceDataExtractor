from transformers import AutoModelForTokenClassification, pipeline

def load_ner_model(model_checkpoint, num_labels, id2label, label2id):
    """
    Loads a Hugging Face AutoModelForTokenClassification for NER.

    Args:
        model_checkpoint (str): Model checkpoint name or path.
        num_labels (int): Number of unique NER labels.
        id2label (dict): Mapping from label ID to label string.
        label2id (dict): Mapping from label string to label ID.

    Returns:
        model: Loaded Hugging Face model for token classification.
    """
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

def save_ner_model(model, tokenizer, save_dir):
    """
    Saves the model and tokenizer to the specified directory.

    Args:
        model: Hugging Face model to save.
        tokenizer: Corresponding tokenizer to save.
        save_dir (str): Directory path to save the model and tokenizer.
    """
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")

def get_ner_pipeline(model, tokenizer, aggregation_strategy="simple", device=-1):
    """
    Returns a Hugging Face pipeline for NER inference.

    Args:
        model: Trained Hugging Face model.
        tokenizer: Corresponding tokenizer.
        aggregation_strategy (str): How to aggregate subword predictions ("simple" recommended).
        device (int): Device to run inference on (-1 for CPU, >=0 for GPU).

    Returns:
        nlp_pipeline: Hugging Face token-classification pipeline.
    """
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy,
        device=device
    )