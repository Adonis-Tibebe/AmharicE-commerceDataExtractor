from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification
from ner_utils import tokenize_and_align_labels, compute_metrics
from ner_models import load_ner_model

def train_model(model_checkpoint, raw_dataset, label2id, id2label, unique_labels, metric, return_model=True):
    """
    Fine-tunes a token classification model using the provided dataset and configuration.

    Args:
        model_checkpoint (str): Model checkpoint name or path.
        raw_dataset (DatasetDict): Hugging Face DatasetDict with 'train' and 'test' splits.
        label2id (dict): Mapping from label string to integer ID.
        id2label (dict): Mapping from integer ID to label string.
        unique_labels (list): List of label strings, indexed by label ID.
        metric: HuggingFace evaluate metric object (e.g., seqeval).

    Returns:
        dict: Evaluation results from Trainer.evaluate().
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
    )

    model = load_ner_model(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="/kaggle/Ouput",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        num_train_epochs=35,
        weight_decay=0.01,   # No external logging
        label_names=["labels"],
        logging_strategy="steps",
        logging_steps = 10,
        report_to=[],
        save_total_limit=0,
        lr_scheduler_type="linear", # Gradually decreases learning rate over training
        warmup_steps = 10 # Increases learning rate for the first 15 steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding='longest',
            label_pad_token_id=-100
        ),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, unique_labels, metric),
    )
    trainer.train()
    metrics = trainer.evaluate()
    if return_model:
        return metrics, model, tokenizer
    return metrics