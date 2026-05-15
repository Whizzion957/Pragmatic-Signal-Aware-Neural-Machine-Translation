"""
rl_trainer.py — Reinforcement Learning from Pragmatic Feedback (RLPF)
======================================================================
REINFORCE-style policy gradient fine-tuning where the reward signal
is the Pragmatic Divergence Score (PDS) between the source and the
generated translation. Inspired by RLHF but uses automated pragmatic
reward instead of human feedback.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── constants ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ─── pragmatic reward function ─────────────────────────────────────
class PragmaticRewardFunction:
    """
    Computes reward for a generated translation as the cosine similarity
    between source and translation in multilingual embedding space.
    Higher PDS = better pragmatic preservation = higher reward.
    """
    def __init__(self, embed_model=None):
        if embed_model is None:
            print(f"📦 Loading reward embedding model: {EMBEDDING_MODEL}")
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        else:
            self.embed_model = embed_model

    def compute_reward(self, sources: list, translations: list) -> torch.Tensor:
        """Compute PDS reward for a batch of source-translation pairs."""
        src_emb = self.embed_model.encode(sources, show_progress_bar=False,
                                          convert_to_tensor=True)
        tgt_emb = self.embed_model.encode(translations, show_progress_bar=False,
                                          convert_to_tensor=True)
        # Cosine similarity as reward
        rewards = F.cosine_similarity(src_emb, tgt_emb, dim=-1)
        return rewards


# ─── REINFORCE trainer ─────────────────────────────────────────────
class RLPragmaticTrainer:
    """
    REINFORCE-based fine-tuning for pragmatic NMT.

    After standard supervised fine-tuning, this trainer applies policy
    gradient updates where:
    - Action = generating a translation token-by-token
    - Reward = PDS(source, generated_translation)
    - Baseline = running mean of rewards (variance reduction)
    - KL penalty = prevents the RL model from drifting too far from
                   the supervised model
    """
    def __init__(
        self,
        model: MarianMTModel,
        tokenizer: MarianTokenizer,
        reward_fn: PragmaticRewardFunction,
        ref_model: MarianMTModel = None,
        learning_rate: float = 1e-6,
        kl_coeff: float = 0.05,
        max_gen_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.lr = learning_rate
        self.kl_coeff = kl_coeff
        self.max_gen_length = max_gen_length
        self.device = next(model.parameters()).device

        # Reference model for KL penalty (frozen copy of supervised model)
        if ref_model is None:
            import copy
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9

    def _sample_translations(self, source_texts: list):
        """
        Sample translations from the policy (model) using multinomial
        sampling (not greedy/beam) — required for REINFORCE.
        Returns decoded texts and per-token log probabilities.
        """
        inputs = self.tokenizer(
            source_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self.device)

        # Use sampling with temperature
        with torch.no_grad():
            sample_outputs = self.model.generate(
                **inputs,
                max_length=self.max_gen_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = sample_outputs.sequences
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Compute log probs under current policy
        # NOTE: we do NOT pass labels= because transformers' Marian uses
        # .view() internally which crashes on non-contiguous tensors.
        # Instead we pass decoder_input_ids and compute log-probs ourselves.
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(generated_ids)
        with torch.enable_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

        # Per-sequence log prob
        logits = outputs.logits  # (batch, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for the generated tokens
        # Shift: logits predict next token
        shift_logits = log_probs[:, :-1, :].contiguous()
        shift_labels = generated_ids[:, 1:].contiguous()
        per_token_lp = torch.gather(
            shift_logits, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        pad_mask = (shift_labels != self.tokenizer.pad_token_id).float()
        seq_log_probs = (per_token_lp * pad_mask).sum(dim=-1) / pad_mask.sum(dim=-1).clamp(min=1)

        return decoded, seq_log_probs, generated_ids

    def _compute_kl_penalty(self, input_ids, attention_mask, generated_ids):
        """
        Compute approximate KL divergence between current policy and
        reference (supervised) model to prevent reward hacking.
        """
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(generated_ids)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
        cur_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        ref_logits = ref_outputs.logits
        cur_logits = cur_outputs.logits

        ref_lp = F.log_softmax(ref_logits, dim=-1)
        cur_lp = F.log_softmax(cur_logits, dim=-1)

        # KL(π_current || π_ref) per token, averaged
        kl = (cur_lp.exp() * (cur_lp - ref_lp)).sum(dim=-1)
        pad_mask = (generated_ids[:, 1:] != self.tokenizer.pad_token_id).float()

        # Align shapes
        kl = kl[:, :-1]
        kl_per_seq = (kl * pad_mask).sum(dim=-1) / pad_mask.sum(dim=-1).clamp(min=1)
        return kl_per_seq.mean()

    def train_step(self, source_texts: list):
        """
        Single REINFORCE training step:
        1. Sample translations from policy
        2. Compute pragmatic reward
        3. Compute advantage (reward - baseline)
        4. Policy gradient: -advantage * log_prob + KL penalty
        """
        self.model.train()

        # 1. Sample translations
        decoded, seq_log_probs, generated_ids = self._sample_translations(source_texts)

        # 2. Compute rewards
        rewards = self.reward_fn.compute_reward(source_texts, decoded)
        rewards = rewards.to(self.device)

        # 3. Update baseline and compute advantage
        mean_reward = rewards.mean().item()
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * mean_reward
        )
        advantages = rewards - self.reward_baseline

        # 4. Policy gradient loss
        pg_loss = -(advantages.detach() * seq_log_probs).mean()

        # 5. KL penalty
        inputs = self.tokenizer(
            source_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self.device)
        kl_penalty = self._compute_kl_penalty(
            inputs["input_ids"], inputs["attention_mask"], generated_ids
        )

        # 6. Total loss
        loss = pg_loss + self.kl_coeff * kl_penalty

        # 7. Backward + update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": mean_reward,
            "baseline": self.reward_baseline,
        }


# ─── high-level training function ─────────────────────────────────
def rl_finetune(
    model,
    tokenizer,
    train_sources: list,
    sarcasm_labels: list = None,
    embed_model=None,
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-6,
    kl_coeff: float = 0.05,
    save_dir: str = "./rl_pragmatic_nmt",
    focus_sarcastic: bool = True,
):
    """
    Run RLPF fine-tuning on top of the pragmatic NMT model.

    Args:
        model: pre-trained pragmatic NMT model
        tokenizer: corresponding tokenizer
        train_sources: list of source sentences (with control tokens)
        sarcasm_labels: optional list of labels (1=sarcastic, 0=literal)
        embed_model: sentence-transformers model for reward
        num_epochs: number of RL epochs
        batch_size: batch size for sampling
        learning_rate: RL learning rate (typically very small)
        kl_coeff: KL penalty coefficient
        save_dir: where to save the RL-tuned model
        focus_sarcastic: if True, oversample sarcastic examples (2x)
    """
    print("🧠 Starting RLPF (RL from Pragmatic Feedback) fine-tuning...")
    print(f"   Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"   KL coeff: {kl_coeff} | Focus sarcastic: {focus_sarcastic}")

    # Prepare training data (oversample sarcastic if requested)
    rl_sources = list(train_sources)
    if focus_sarcastic and sarcasm_labels is not None:
        sarc_sources = [s for s, l in zip(train_sources, sarcasm_labels) if l == 1]
        rl_sources.extend(sarc_sources)  # 2x sarcastic examples
        print(f"   Training examples: {len(rl_sources)} (with sarcastic oversampling)")

    # Initialize reward function and trainer
    reward_fn = PragmaticRewardFunction(embed_model)
    trainer = RLPragmaticTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        learning_rate=learning_rate,
        kl_coeff=kl_coeff,
    )

    # Training loop
    history = []
    for epoch in range(num_epochs):
        random.shuffle(rl_sources)
        epoch_stats = {"loss": [], "pg_loss": [], "kl_penalty": [], "mean_reward": []}

        pbar = tqdm(
            range(0, len(rl_sources), batch_size),
            desc=f"RL Epoch {epoch+1}/{num_epochs}",
        )
        for i in pbar:
            batch = rl_sources[i:i+batch_size]
            if len(batch) < 2:
                continue

            # Strip control tokens for reward computation
            clean_batch = [
                s.replace("<SARCASTIC> ", "").replace("<LITERAL> ", "")
                for s in batch
            ]

            stats = trainer.train_step(batch)
            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

            pbar.set_postfix({
                "reward": f"{stats['mean_reward']:.4f}",
                "loss": f"{stats['loss']:.4f}",
            })

        # Epoch summary
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        history.append(avg_stats)
        print(f"   Epoch {epoch+1}: reward={avg_stats['mean_reward']:.4f} | "
              f"pg_loss={avg_stats['pg_loss']:.4f} | kl={avg_stats['kl_penalty']:.4f}")

    # Save the RL-tuned model
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"   ✅ RL-tuned model saved to {save_dir}")

    return model, history
