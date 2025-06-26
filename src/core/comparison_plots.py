import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from transformers import Trainer, DataCollatorForTokenClassification

def plot_confusion(model_name, model, tokenizer, dataset, label2id, id2label):
    """
    Plots the confusion matrix for a given model and dataset.
    """
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest',
        label_pad_token_id=-100
    ))

    predictions, labels, _ = trainer.predict(dataset)
    preds = np.argmax(predictions, axis=-1)

    # Flatten and remove -100s
    true_labels = []
    pred_labels = []
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(label2id.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label2id.keys()))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(include_values=True, xticks_rotation="vertical", ax=ax)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

def plot_multiclass_roc(model_name, model, tokenizer, dataset, label2id):
    """
    Compute and plot ROC curves for each class in a multilabel setting.
    """
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest',
        label_pad_token_id=-100
    ))
    predictions, labels, _ = trainer.predict(dataset)
    
    import torch
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()

    # Flatten predictions and labels
    mask = labels != -100
    true_labels = labels[mask]
    probs = probs[mask]

    # Binarize true labels for ROC
    true_bin = label_binarize(true_labels, classes=list(label2id.values()))

    plt.figure(figsize=(10, 7))
    for i, class_id in enumerate(label2id.values()):
        fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{list(label2id.keys())[i]} (AUC = {roc_auc:.2f})")

    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()