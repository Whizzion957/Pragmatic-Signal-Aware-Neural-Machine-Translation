"""
data.py — Dataset Loading and Preprocessing (v3)
==================================================
Loads the irony dataset and IITB En-Hi parallel corpus. Creates a
sarcasm-enriched parallel corpus with separate IITB test set for
fair BLEU evaluation.
"""

import re
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Value, ClassLabel

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─── text preprocessing ────────────────────────────────────────────
def preprocess_tweet(text: str) -> str:
    """Clean a tweet while keeping punctuation (important for sarcasm)."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── sarcasm dataset ───────────────────────────────────────────────
def load_sarcasm_dataset():
    """Load tweet_eval/irony dataset."""
    print("📥 Loading irony/sarcasm dataset (tweet_eval/irony)...")
    ds = load_dataset("tweet_eval", "irony")

    def _clean(example):
        example["text"] = preprocess_tweet(example["text"])
        return example

    ds = ds.map(_clean)
    train_ds = ds["train"]
    if "validation" in ds:
        train_ds = concatenate_datasets([ds["train"], ds["validation"]])

    return DatasetDict({"train": train_ds, "test": ds["test"]})


# ─── parallel English-Hindi corpus ─────────────────────────────────
def load_parallel_dataset(num_samples: int = 10_000, test_size: float = 0.1):
    """Load IITB English-Hindi parallel dataset."""
    print(f"📥 Loading IITB En-Hi parallel dataset ({num_samples} pairs)...")
    ds = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)

    en_texts, hi_texts = [], []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        translation = example["translation"]
        en = preprocess_tweet(translation["en"])
        hi = translation["hi"]
        if len(en.split()) >= 3 and len(hi.split()) >= 3:
            en_texts.append(en)
            hi_texts.append(hi)

    print(f"   ✅ Loaded {len(en_texts)} valid sentence pairs")

    full_ds = Dataset.from_dict({"en": en_texts, "hi": hi_texts})
    split = full_ds.train_test_split(test_size=test_size, seed=SEED)
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ─── sarcasm-enriched parallel corpus ──────────────────────────────
def create_sarcasm_enriched_parallel(
    sarcasm_ds,
    base_nmt_model,
    base_nmt_tokenizer,
    max_samples: int = 2000,
    batch_size: int = 32,
):
    """
    Generate synthetic sarcastic En-Hi parallel pairs from irony tweets.
    Uses baseline MarianMT for silver-standard translations.
    """
    print("🔧 Creating sarcasm-enriched parallel corpus...")

    sarc_texts = [
        ex["text"] for ex in sarcasm_ds["train"] if ex["label"] == 1
    ][:max_samples]

    lit_texts = [
        ex["text"] for ex in sarcasm_ds["train"] if ex["label"] == 0
    ][:max_samples]

    print(f"   Sarcastic tweets:  {len(sarc_texts)}")
    print(f"   Literal tweets:    {len(lit_texts)}")

    device = next(base_nmt_model.parameters()).device

    def translate_batch_texts(texts, bsz=batch_size):
        translations = []
        for i in range(0, len(texts), bsz):
            batch = texts[i:i+bsz]
            inputs = base_nmt_tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(device)
            with torch.no_grad():
                outputs = base_nmt_model.generate(
                    **inputs, max_length=128, num_beams=4,
                )
            decoded = base_nmt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
        return translations

    print("   Translating sarcastic tweets to Hindi...")
    sarc_hi = translate_batch_texts(sarc_texts)
    print("   Translating literal tweets to Hindi...")
    lit_hi = translate_batch_texts(lit_texts)

    all_en = sarc_texts + lit_texts
    all_hi = sarc_hi + lit_hi
    all_labels = [1] * len(sarc_texts) + [0] * len(lit_texts)
    all_tokens = ["<SARCASTIC>"] * len(sarc_texts) + ["<LITERAL>"] * len(lit_texts)

    ds = Dataset.from_dict({
        "en": all_en, "hi": all_hi,
        "sarcasm_label": all_labels, "control_token": all_tokens,
    })
    ds = ds.shuffle(seed=SEED)
    n_sarc = sum(all_labels)
    print(f"   ✅ Created {len(ds)} enriched pairs ({n_sarc} sarcastic, {len(ds)-n_sarc} literal)")
    return ds


def build_combined_dataset(
    iitb_parallel_ds,
    sarcasm_enriched_ds,
    sarcasm_pipeline,
    batch_size: int = 64,
    test_size: float = 0.1,
):
    """
    Combine IITB + enriched data for training.
    Test set from enriched data for sentiment/PDS evaluation.
    IITB test kept separate for fair BLEU evaluation.
    """
    print("📦 Building combined dataset...")

    # Pseudo-label the IITB data
    def _label_batch(batch):
        preds = sarcasm_pipeline(batch["en"], batch_size=batch_size, truncation=True)
        labels, tokens = [], []
        for pred in preds:
            lbl = pred["label"].upper()
            is_sarc = lbl in ("LABEL_1", "IRONY", "SARCASTIC", "1")
            labels.append(1 if is_sarc else 0)
            tokens.append("<SARCASTIC>" if is_sarc else "<LITERAL>")
        return {"sarcasm_label": labels, "control_token": tokens}

    print("   Pseudo-labelling IITB data...")
    iitb_labelled = iitb_parallel_ds["train"].map(
        _label_batch, batched=True, batch_size=batch_size
    )

    # Split enriched into train/test
    enriched_with_cl = sarcasm_enriched_ds.cast_column(
        "sarcasm_label", ClassLabel(names=["literal", "sarcastic"])
    )
    enriched_split = enriched_with_cl.train_test_split(
        test_size=test_size, seed=SEED, stratify_by_column="sarcasm_label"
    )
    for split_name in enriched_split:
        enriched_split[split_name] = enriched_split[split_name].cast_column(
            "sarcasm_label", Value("int64")
        )

    # Combine for training
    combined_train = concatenate_datasets([iitb_labelled, enriched_split["train"]])
    combined_train = combined_train.shuffle(seed=SEED)
    test_ds = enriched_split["test"]

    train_sarc = sum(combined_train["sarcasm_label"])
    test_sarc = sum(test_ds["sarcasm_label"])
    print(f"   Train: {len(combined_train)} ({train_sarc} sarcastic, {len(combined_train)-train_sarc} literal)")
    print(f"   Test:  {len(test_ds)} ({test_sarc} sarcastic, {len(test_ds)-test_sarc} literal)")

    return DatasetDict({"train": combined_train, "test": test_ds})


def prepend_control_token(example):
    """Prepend control token to the English source text."""
    example["en_with_token"] = example["control_token"] + " " + example["en"]
    return example


if __name__ == "__main__":
    print("=" * 60)
    print("Testing data.py standalone")
    print("=" * 60)
    sarcasm_ds = load_sarcasm_dataset()
    print(f"\nSarcasm dataset: Train={len(sarcasm_ds['train'])}, Test={len(sarcasm_ds['test'])}")
    parallel_ds = load_parallel_dataset(num_samples=100)
    print(f"Parallel dataset: Train={len(parallel_ds['train'])}, Test={len(parallel_ds['test'])}")
