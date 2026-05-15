"""
contrastive.py — Contrastive Pragmatic Embedding Alignment
===========================================================
InfoNCE-based contrastive learning objective that aligns the encoder
representations of sarcastic source sentences with their translations,
while pushing apart sarcastic-literal pairs. Forces the NMT model's
internal representations to be pragmatically aware.

Inspired by SimCLR/CLIP but applied to pragmatic signal alignment
in machine translation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from tqdm.auto import tqdm

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─── InfoNCE / NT-Xent loss ───────────────────────────────────────
class PragmaticContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for pragmatic alignment.

    Positive pairs: (source_sarcastic, translation_sarcastic)
    Negative pairs: (source_sarcastic, translation_literal) and vice versa

    This forces the encoder to produce embeddings that capture pragmatic
    intent, not just surface semantics.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, source_embeds: torch.Tensor, target_embeds: torch.Tensor,
                sarcasm_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_embeds: (batch, hidden_dim) — encoder output for source
            target_embeds: (batch, hidden_dim) — encoder output for target
            sarcasm_labels: (batch,) — 1 for sarcastic, 0 for literal
        """
        # Normalize embeddings
        source_embeds = F.normalize(source_embeds, dim=-1)
        target_embeds = F.normalize(target_embeds, dim=-1)

        batch_size = source_embeds.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=source_embeds.device, requires_grad=True)

        # Compute similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(source_embeds, target_embeds.T) / self.temperature

        # Positive pairs: diagonal (source_i matches with target_i)
        # But we also want pragmatically similar pairs to be closer
        # Create soft positive mask: same sarcasm label = positive
        labels_row = sarcasm_labels.unsqueeze(1)  # (batch, 1)
        labels_col = sarcasm_labels.unsqueeze(0)  # (1, batch)
        same_label_mask = (labels_row == labels_col).float()  # (batch, batch)

        # Ensure diagonal is always positive
        eye = torch.eye(batch_size, device=source_embeds.device)
        positive_mask = torch.clamp(same_label_mask + eye, max=1.0)

        # InfoNCE: for each source, maximize similarity with its own
        # translation and other same-label translations
        log_softmax = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Average log-prob over positive pairs
        num_positives = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(log_softmax * positive_mask).sum(dim=1) / num_positives
        return loss.mean()


# ─── multi-task contrastive + translation trainer ──────────────────
class ContrastiveNMTTrainer(Seq2SeqTrainer):
    """
    Multi-task trainer combining:
    1. Standard NMT translation loss (cross-entropy)
    2. Contrastive pragmatic alignment loss (InfoNCE)
    3. Sarcasm-weighted loss (from translator.py)

    Total loss = α * NMT_loss + β * contrastive_loss + γ * sarcasm_weight
    """
    def __init__(self, *args, contrastive_weight: float = 0.1,
                 sarcasm_loss_weight: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = PragmaticContrastiveLoss(temperature=0.07)
        self.contrastive_weight = contrastive_weight
        self.sarcasm_loss_weight = sarcasm_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sarcasm_weights = inputs.pop("sarcasm_weight", None)
        sarcasm_labels = inputs.pop("sarcasm_labels_for_cl", None)

        outputs = model(**inputs)

        # ── 1. Sarcasm-weighted NMT loss ────────────────────────────
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
            nmt_loss = (per_example_loss * weights).mean()
        else:
            nmt_loss = outputs.loss

        # ── 2. Contrastive loss ─────────────────────────────────────
        contrastive_loss = torch.tensor(0.0, device=nmt_loss.device)
        if sarcasm_labels is not None and len(sarcasm_labels) >= 4:
            # Get encoder hidden states (mean pool over sequence)
            encoder_outputs = model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            src_embeds = encoder_outputs.last_hidden_state
            src_mask = inputs["attention_mask"].unsqueeze(-1).float()
            src_pooled = (src_embeds * src_mask).sum(dim=1) / src_mask.sum(dim=1).clamp(min=1)

            # Get decoder hidden states for targets (use encoder on target tokens)
            # We'll use the decoder's last hidden state as target embedding
            decoder_hidden = outputs.decoder_hidden_states[-1] if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states else None

            if decoder_hidden is not None:
                tgt_mask = (inputs["labels"] != -100).unsqueeze(-1).float()
                tgt_pooled = (decoder_hidden * tgt_mask).sum(dim=1) / tgt_mask.sum(dim=1).clamp(min=1)
            else:
                # Fallback: use logits mean as target representation
                tgt_pooled = outputs.logits.mean(dim=1)

            # Project to same dimension if needed
            if src_pooled.size(-1) != tgt_pooled.size(-1):
                tgt_pooled = tgt_pooled[:, :src_pooled.size(-1)]

            contrastive_loss = self.contrastive_loss_fn(
                src_pooled, tgt_pooled, sarcasm_labels.to(src_pooled.device)
            )

        # ── 3. Total loss ───────────────────────────────────────────
        total_loss = nmt_loss + self.contrastive_weight * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ─── high-level training function ─────────────────────────────────
def train_with_contrastive(
    train_dataset,
    eval_dataset,
    tokenizer,
    model,
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    contrastive_weight: float = 0.1,
    save_dir: str = "./contrastive_pragmatic_nmt",
):
    """
    Fine-tune NMT with contrastive pragmatic alignment.

    This adds an InfoNCE contrastive objective on top of the standard
    translation loss, forcing the model to learn pragmatically-aware
    internal representations.
    """
    print("🧠 Fine-tuning with Contrastive Pragmatic Alignment...")
    print(f"   Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"   Contrastive weight: {contrastive_weight}")

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
        generation_max_length=128,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = ContrastiveNMTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        contrastive_weight=contrastive_weight,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"   ✅ Contrastive model saved to {save_dir}")
    return trainer
