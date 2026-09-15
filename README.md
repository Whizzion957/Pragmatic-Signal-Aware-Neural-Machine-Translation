# Pragmatic Signal-Aware Neural Machine Translation
 
**Preserving Sarcasm in English-to-Hindi Social Media Translation**
 
> NLP Laboratory Project — Department of Computer Science & Engineering, Indian Institute of Technology Roorkee
 
Standard NMT systems optimize for lexical and syntactic fidelity but systematically discard pragmatic signals — sarcasm in particular — that are critical to communicative intent. This project introduces a multi-stage pipeline for pragmatic-aware English-to-Hindi translation of social media text, combining fine-tuned sarcasm detection, a synthetic sarcasm-enriched parallel corpus, control-token conditioned NMT with weighted loss, contrastive pragmatic alignment, and Reinforcement Learning from Pragmatic Feedback (RLPF).
 
On sarcastic social media content, the best model achieves a **+6.84%** improvement in Sentiment Preservation Rate and **+9.86 BLEU points** over the baseline, while the RL-enhanced variant resolves hard cases where both the baseline and pragmatic models fail.
 
---
 
## Table of Contents
 
- [Motivation](#motivation)
- [Pipeline Overview](#pipeline-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Novel Contributions](#novel-contributions)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
---
 
## Motivation
 
A user tweets *"oh great, another Monday morning, just what I needed."* Every English speaker recognizes the sarcasm immediately, but a standard translation system renders this as a genuinely enthusiastic sentence in Hindi — inverting the intended meaning.
 
This isn't an edge case. Sarcasm is pervasive in social media, and downstream applications — sentiment analysis, content moderation, cross-lingual opinion mining — all depend on pragmatic fidelity surviving translation. Existing NMT systems (Google Translate, MarianMT, mBART) are evaluated on BLEU, a metric that counts matching word sequences, with no mechanism to detect or preserve pragmatic signals.
 
This work is, to our knowledge, the first to combine sarcasm-conditioned control tokens, contrastive representation alignment, and pragmatic RL fine-tuning for NMT — all built on MarianMT (~74M parameters), making the pipeline reproducible on a single free Colab T4 GPU in under an hour per stage. The contribution is the technique, not the base model — it is architecture-agnostic and transferable to mBART, NLLB, or any future system.
 
## Pipeline Overview
 
```
English Tweet
     │
     ▼
Sarcasm Detector (twitter-roberta-base)
     │
     ▼
<SARCASTIC> or <LITERAL> prefix
     │
     ▼
Pragmatic-Aware MarianMT
(control tokens + weighted loss
 + contrastive alignment + RL fine-tuning)
     │
     ▼
Hindi Output (with preserved pragmatic tone)
```
 
## Methodology
 
The pipeline comprises five stages; Stages 3–5 are the primary novel contributions.
 
### Stage 1 — Sarcasm Detection
Fine-tuned `twitter-roberta-base` (RoBERTa pre-trained on 58M tweets) on the `tweet_eval/irony` dataset (3,817 tweets, 49.8% sarcastic). 3 epochs, batch size 16, LR 2×10⁻⁵, Colab T4 GPU.
 
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Sarcastic | 0.87 | 0.75 | 0.81 |
| Sarcastic | 0.69 | 0.83 | 0.75 |
| **Overall Accuracy** | | | **78.19%** |
 
### Stage 2 — Sarcasm-Enriched Parallel Corpus *(Novel)*
Standard parallel corpora (e.g., IITB English-Hindi) contain formal text with virtually no sarcasm; sarcasm datasets are English-only with no Hindi translations. We bridge this gap by:
- Extracting sarcastic tweets from the detection dataset
- Generating "silver-standard" Hindi translations with the base MarianMT model
- Tagging each pair with a binary sarcasm label
- Combining with 3,985 IITB pairs (up to 2,000 sarcasm-enriched pairs added)
### Stage 3 — Control Token + Weighted Loss NMT *(Novel)*
Extends `Helsinki-NLP/opus-mt-en-hi` with:
- **Control tokens** — `<SARCASTIC>` / `<LITERAL>` prepended to input, conditioning the decoder on pragmatic type without a second model
- **Sarcasm-weighted loss** — loss on sarcastic examples scaled by λ = 1.5 to prevent the minority class from being overwhelmed (λ = 3.0 caused quality collapse)
3 epochs, batch size 8, LR 3×10⁻⁵, FP16 mixed precision.
 
### Stage 4 — Contrastive Pragmatic Alignment *(Novel)*
An InfoNCE contrastive loss (inspired by CLIP/SimCLR) applied to the encoder's internal representations:
 
```
L_CPA = -log [ exp(sim(h_s, h_t+)/τ) / Σ_k exp(sim(h_s, h_k)/τ) ]
```
 
Positive pairs: sarcastic source + its target translation. Negative pairs: sarcastic source + literal translations. Total loss `L = L_NMT + α·L_CPA` with α = 0.1. Forces the encoder to develop geometrically distinct representations for sarcastic vs. literal content. 2 epochs.
 
### Stage 5 — RL from Pragmatic Feedback (RLPF) *(Novel)*
Inspired by RLHF. Uses an automated **Pragmatic Divergence Score (PDS)** — cosine similarity between English and Hindi sentence embeddings (`paraphrase-multilingual-MiniLM`) — as the reward, instead of human feedback.
 
A REINFORCE policy-gradient update generates a translation as the "action," rewarded by PDS, with a KL penalty (β = 0.05) against reward hacking and a running baseline to reduce variance:
 
```
∇L_RLPF = -E[(r - b)·∇log π_θ(ŷ|x)] + β·D_KL(π_θ‖π_ref)
```
 
1 epoch, LR 1×10⁻⁶, batch size 8.
 
## Results
 
| Metric | Baseline | Pragmatic | RL |
|---|---|---|---|
| BLEU (IITB) | 64.82 | **74.68** | 70.43 |
| Sentiment Preservation (%) ↑ | 54.21 | **61.05** | 59.47 |
| PDS ↑ | 0.6877 | **0.7037** | 0.5819 |
| Sentiment Divergence ↓ | 0.1615 | **0.1553** | 0.1618 |
 
The **Pragmatic-Aware model (Stage 4)** is the aggregate winner, improving every metric simultaneously — including a **+9.86 BLEU** gain, showing that sarcasm-awareness generalizes to improve translation quality even on literal text.
 
The **RL-Enhanced model** trades a lower aggregate PDS (occasional repetitive outputs on complex sentences) for decisive superiority on hard cases where the other two models fail:
 
> *"i love how when i'm stressed my body decides to react by causing me massive pain."*
 
| Model | Predicted Sentiment | Correct? |
|---|---|---|
| Baseline | Negative | ✗ |
| Pragmatic | Negative | ✗ |
| **RL-Enhanced** | **Positive** | **✓** |
 
The RL model learns subtle lexical choices — e.g., preserving "love" as an intensifier rather than genuine affection — that flip the translated sentiment to match the source: the operational definition of pragmatic preservation.
 
## Novel Contributions
 
1. **Sarcasm-Enriched Corpus** — synthetic parallel data with pragmatic labels, bridging English-only sarcasm datasets and label-free parallel corpora.
2. **Control Token + Weighted Loss NMT** — pragmatic-type conditioning via special tokens combined with minority-class loss amplification (λ = 1.5).
3. **Contrastive Pragmatic Alignment** — InfoNCE loss applied to NMT encoder representations for geometrically distinct sarcasm/literal structure.
4. **RLPF + PDS** — REINFORCE policy gradient using multilingual cosine similarity as reward, optimizing directly for pragmatic fidelity rather than surface word overlap.
The Pragmatic Divergence Score (PDS) is itself a contribution: a language-agnostic, meaning-level translation quality metric sensitive to tone and pragmatic register, usable both as an evaluation metric and as an RL reward signal.
 
## Evaluation Metrics
 
**Standard MT:**
- BLEU (n-gram overlap)
- BERTScore (contextual semantic similarity)
- chrF++ (character-level, better suited to Hindi morphology)
**Pragmatic (novel):**
- **Sentiment Preservation Rate** — match rate between source and translation sentiment classification
- **Pragmatic Divergence Score (PDS)** — multilingual cosine similarity (higher = better)
- **Sentiment Divergence** — KL divergence between sentiment distributions (lower = better)
## Limitations & Future Work
 
- The enriched corpus uses machine-generated Hindi as reference (silver standard), introducing noise; human translation references would improve quality.
- The RL model's aggregate PDS regression suggests instability on longer, complex sentences — a curriculum strategy with more epochs could address this.
- The sarcasm detector operates at 78% F1; misclassifications propagate noise through the pipeline.
**Future directions:** human evaluation studies; extension to irony, understatement, and rhetorical questions; application to other language pairs (politeness in Japanese-Korean, register in Hindi-Urdu); scaling to mBART or NLLB as the base model.
 
## References
 
1. Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692, 2019.
2. Barbieri et al. *TweetEval: Unified Benchmark for Tweet Classification.* EMNLP Findings, 2020.
3. Junczys-Dowmunt et al. *Marian: Fast Neural Machine Translation in C++.* ACL System Demonstrations, 2018.
4. Kunchukuttan et al. *The IIT Bombay English-Hindi Parallel Corpus.* LREC, 2018.
5. Papineni et al. *BLEU: A Method for Automatic Evaluation of Machine Translation.* ACL, 2002.
6. Zhang et al. *BERTScore: Evaluating Text Generation with BERT.* ICLR, 2020.
7. Reimers & Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP, 2019.
8. van den Oord et al. *Representation Learning with Contrastive Predictive Coding (InfoNCE).* arXiv:1807.03748, 2018.
9. Williams, R. J. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE).* Machine Learning, 1992.
10. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback (RLHF).* NeurIPS, 2022.
---
 
*Author: Aadit Kumar Sahoo · NLP Lab, Department of Computer Science & Engineering, IIT Roorkee*
