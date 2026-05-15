"""
translator.py — Control-Token-Injected NMT Module (v3)
=======================================================
Fine-tunes Helsinki-NLP/opus-mt-en-hi with control tokens and a
tuned sarcasm-weighted loss (1.5x, not 3x to avoid quality collapse).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── constants ──────────────────────────────────────────────────────
BASE_MODEL = "Helsinki-NLP/opus-mt-en-hi"
SAVE_DIR = "./pragmatic_nmt"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128
CONTROL_TOKENS = ["<SARCASTIC>", "<LITERAL>"]
SARCASM_LOSS_WEIGHT = 1.5  # mild upweight (v2 used 3.0 which collapsed BLEU)


def load_base_model_and_tokenizer():
    """Load the vanilla MarianMT En→Hi model and tokenizer."""
    print(f"📦 Loading base NMT model: {BASE_MODEL}")
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL)
    model = MarianMTModel.from_pretrained(BASE_MODEL)
    return model, tokenizer


def load_pragmatic_model_and_tokenizer(model_dir: str = SAVE_DIR):
    """Load the fine-tuned pragmatic NMT model."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Pragmatic NMT not found at {model_dir}")
    print(f"📦 Loading pragmatic NMT model from {model_dir}")
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    return model, tokenizer


def add_control_tokens(tokenizer, model):
    """Add control tokens to the tokenizer and resize model embeddings."""
    num_added = tokenizer.add_tokens(CONTROL_TOKENS, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"   ✅ Added {num_added} control tokens to tokenizer")
    return tokenizer, model


def prepare_nmt_dataset(parallel_ds, tokenizer):
    """Tokenize parallel dataset for Seq2Seq training with sarcasm weights."""
    def _tokenize(examples):
        model_inputs = tokenizer(
            examples["en_with_token"],
            max_length=MAX_SOURCE_LEN,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["hi"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in model_inputs["labels"]
        ]
        model_inputs["sarcasm_weight"] = [
            SARCASM_LOSS_WEIGHT if sl == 1 else 1.0
            for sl in examples["sarcasm_label"]
        ]
        return model_inputs

    cols_to_remove = [c for c in parallel_ds.column_names if c not in ("sarcasm_weight",)]
    return parallel_ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)


class SarcasmWeightedTrainer(Seq2SeqTrainer):
    """Custom trainer with per-example sarcasm loss weighting."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sarcasm_weights = inputs.pop("sarcasm_weight", None)
        outputs = model(**inputs)

        if sarcasm_weights is not None:
            logits = outputs.logits
            labels = inputs["labels"]
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1)
            per_token_loss = loss_fct(shift_logits, shift_labels)
            per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))
            mask = (labels != -100).float()
            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            weights = sarcasm_weights.float().to(per_example_loss.device)
            loss = (per_example_loss * weights).mean()
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def train_pragmatic_nmt(
    train_dataset, eval_dataset, tokenizer, model,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    save_dir: str = SAVE_DIR,
    use_weighted_loss: bool = True,
):
    """Fine-tune MarianMT with control tokens and sarcasm-weighted loss."""
    print("🧠 Fine-tuning Pragmatic NMT model (v3)...")
    print(f"   Base: {BASE_MODEL}")
    print(f"   Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"   Weighted loss: {use_weighted_loss} (weight: {SARCASM_LOSS_WEIGHT}x)")

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    TrainerClass = SarcasmWeightedTrainer if use_weighted_loss else Seq2SeqTrainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"   ✅ Model saved to {save_dir}")
    return trainer


def translate_batch(texts, model, tokenizer, max_length=MAX_TARGET_LEN):
    """Translate a batch of English texts to Hindi."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SOURCE_LEN,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=max_length, num_beams=4, early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_single(text, model, tokenizer, control_token=None):
    """Translate a single sentence, optionally with control token."""
    if control_token:
        text = f"{control_token} {text}"
    return translate_batch([text], model, tokenizer)[0]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing translator.py standalone")
    print("=" * 60)
    model, tokenizer = load_base_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for sent in ["The weather is beautiful today.", "Oh great, another traffic jam."]:
        print(f"  EN: {sent}")
        print(f"  HI: {translate_single(sent, model, tokenizer)}\n")
