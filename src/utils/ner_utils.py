

def load_conll(path):
    """
    Loads a CoNLL file and returns sentences and their corresponding labels.
    Only extracts tokens and NER tags (ignores POS/Chunk columns).

    Args:
        path (str): Path to the .conll file.

    Returns:
        sentences (List[List[str]]): List of tokenized sentences.
        labels (List[List[str]]): List of NER tag sequences.
    """
    sentences, labels = [], []
    curr_tokens, curr_labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Sentence boundary or DOCSTART marker
            if not line or line.startswith("-DOCSTART-"):
                if curr_tokens:
                    sentences.append(curr_tokens)
                    labels.append(curr_labels)
                    curr_tokens, curr_labels = [], []
            else:
                token, _, _, ner_tag = line.split()
                curr_tokens.append(token)
                curr_labels.append(ner_tag)
        # last sentence
        if curr_tokens:
            sentences.append(curr_tokens)
            labels.append(curr_labels)
    return sentences, labels


def tokenize_and_align_labels(examples, tokenizer, label2id):
    """
    Tokenizes input tokens and aligns NER labels with subword tokens.
    Converts word-level tokens and ner_tags into subword-level inputs that
    XLM-RoBERTa can consume, propagating BIO tags with word_ids().

    Args:
        examples (dict): Dictionary with "tokens" and "ner_tags" keys.
        tokenizer: HuggingFace tokenizer.
        label2id (dict): Mapping from label string to integer ID.

    Returns:
        tokenized_inputs (BatchEncoding): Tokenized inputs with aligned labels.
    """
    # Tokenize the batch of word-lists with is_split_into_words=True
    tokenized_inputs = tokenizer(
        examples["tokens"],            # List of word tokens
        is_split_into_words=True,      # Keep track of word boundaries
        truncation=True,               # Truncate long sequences
        padding=False,                 # No padding here (done during Data collation)
    )
    all_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None  # to capture starting and ending tags (<s> and </s>)
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignore index)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word → original tag id
                label_ids.append(label2id[labels[word_idx]])
            else:
                # Continuation subword → I- prefix (words which were split by tokenizer)
                tag = labels[word_idx]
                if tag.startswith("B-"):
                    tag = tag.replace("B-", "I-")  # replace B- tags with I- tags on non beginning subwords
                label_ids.append(label2id[tag])
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels  # Add the newly aligned labels column
    return tokenized_inputs


def compute_metrics(eval_preds, unique_labels, metric):
    """
    Computes precision, recall, f1, and accuracy for NER predictions.

    Args:
        eval_preds (tuple): Tuple of (logits, labels) from the Trainer.
        unique_labels (list): List of label strings, indexed by label ID.
        metric: HuggingFace evaluate metric object (e.g., seqeval).

    Returns:
        dict: Dictionary with precision, recall, f1, and accuracy.
    """
    import numpy as np
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[unique_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }