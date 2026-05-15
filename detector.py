"""
detector.py — Sarcasm Detection Module
========================================
Fine-tunes cardiffnlp/twitter-roberta-base on the iSarcasm dataset
for binary sarcasm classification. Provides training, evaluation,
and inference utilities.
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from datasets import DatasetDict

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── constants ──────────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
SAVE_DIR = "./sarcasm_detector"
MAX_LENGTH = 128
LABEL2ID = {"not_sarcastic": 0, "sarcastic": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ─── tokenization ──────────────────────────────────────────────────
def get_tokenizer():
    """Load the tokenizer for twitter-roberta-base."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset: DatasetDict, tokenizer=None):
    """Tokenize the sarcasm dataset for training."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized


# ─── metrics ────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute accuracy and F1 for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}


# ─── training ──────────────────────────────────────────────────────
def train_sarcasm_detector(
    train_dataset,
    eval_dataset,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    save_dir: str = SAVE_DIR,
):
    """
    Fine-tune twitter-roberta-base for binary sarcasm classification.
    Returns the trained Trainer object.
    """
    print("🧠 Fine-tuning sarcasm detector...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")

    tokenizer = get_tokenizer()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save best model
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"   ✅ Model saved to {save_dir}")

    return trainer


# ─── evaluation ─────────────────────────────────────────────────────
def evaluate_detector(trainer, test_dataset):
    """
    Run evaluation on the test set and print detailed metrics.
    Returns a dict with accuracy, f1, and the full classification report.
    """
    print("\n📊 Evaluating sarcasm detector on test set...")
    results = trainer.evaluate(test_dataset)

    # Get predictions for detailed report
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    report = classification_report(
        labels, preds,
        target_names=["Not Sarcastic", "Sarcastic"],
        digits=4,
    )
    print(report)

    metrics = {
        "accuracy": results["eval_accuracy"],
        "f1": results["eval_f1"],
        "report": report,
    }
    return metrics


# ─── inference pipeline ────────────────────────────────────────────
def load_sarcasm_pipeline(model_dir: str = SAVE_DIR):
    """Load a saved sarcasm detection model as a HuggingFace pipeline."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Sarcasm detector not found at {model_dir}. Train it first!"
        )
    print(f"📦 Loading sarcasm pipeline from {model_dir}")
    return pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=0 if torch.cuda.is_available() else -1,
        max_length=MAX_LENGTH,
        truncation=True,
    )


def predict_sarcasm(text: str, pipe=None, model_dir: str = SAVE_DIR):
    """
    Predict whether a single text is sarcastic.
    Returns: dict with 'label' (str), 'score' (float), 'is_sarcastic' (bool)
    """
    if pipe is None:
        pipe = load_sarcasm_pipeline(model_dir)

    result = pipe(text)[0]
    is_sarcastic = result["label"] in ("sarcastic", "LABEL_1")

    return {
        "label": "Sarcastic" if is_sarcastic else "Literal",
        "score": result["score"],
        "is_sarcastic": is_sarcastic,
    }


# ─── main (standalone test) ─────────────────────────────────────────
if __name__ == "__main__":
    from data import load_sarcasm_dataset

    print("=" * 60)
    print("Testing detector.py standalone")
    print("=" * 60)

    # Load and tokenize data
    sarcasm_ds = load_sarcasm_dataset()
    tokenizer = get_tokenizer()
    tokenized_ds = tokenize_dataset(sarcasm_ds, tokenizer)

    # Train
    trainer = train_sarcasm_detector(
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
    )

    # Evaluate
    metrics = evaluate_detector(trainer, tokenized_ds["test"])
    print(f"\n✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✅ F1 Score: {metrics['f1']:.4f}")

    # Quick inference test
    pipe = load_sarcasm_pipeline()
    test_texts = [
        "oh great, another monday morning",
        "i love sunny days at the park",
        "wow what a fantastic way to start the day, said no one ever",
    ]
    for t in test_texts:
        r = predict_sarcasm(t, pipe)
        print(f"  '{t}' → {r['label']} ({r['score']:.3f})")
