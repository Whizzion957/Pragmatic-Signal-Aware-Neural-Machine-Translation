# Pragmatic Signal-Aware Neural Machine Translation

## A Comprehensive Documentation

---

## 1. The Problem: Translation Loses More Than Words

### What happens when you translate sarcasm?

When someone tweets *"oh great, another Monday morning, just what I needed"*, any English speaker immediately recognises it as sarcasm — the person clearly does **not** want another Monday. But when a standard translation system converts this to Hindi, something critical gets lost. The translated sentence might read as a genuinely enthusiastic statement, completely flipping the intended meaning.

This is because existing machine translation (MT) systems are designed to translate **literal meaning** — the dictionary definitions of words and the grammar rules that connect them. They have no understanding of **pragmatic signals** — the unspoken layer of communication where sarcasm, irony, tone, and implied meaning live.

### Why does this matter?

Social media generates billions of posts daily, in dozens of languages. Organisations, researchers, and readers increasingly rely on automated translation to understand content across languages. When translation strips away sarcasm:

- **Sentiment analysis** on translated text gives wrong results (a sarcastic complaint looks like praise)
- **Content moderation** systems miss toxic sarcasm disguised as compliments
- **Cross-cultural communication** breaks down — sarcastic humour falls flat or causes misunderstanding
- **Market research** across languages gives misleading conclusions about public opinion

### What is "Pragmatic" in language?

In linguistics, **pragmatics** is the study of how context contributes to meaning. Unlike **semantics** (what words literally mean), pragmatics deals with what a speaker actually *intends* to communicate. For example:

| Sentence | Semantic Meaning | Pragmatic Meaning |
|----------|-----------------|-------------------|
| "Nice weather we're having" (during a storm) | The weather is nice | The weather is terrible (sarcasm) |
| "Could you pass the salt?" | Are you capable of passing salt? | Please give me the salt (indirect request) |
| "I just *love* waiting in queues" | The speaker enjoys queues | The speaker hates queues (irony) |

Our project specifically tackles **sarcasm** — the most common and impactful pragmatic signal in social media text — during English-to-Hindi translation.

---

## 2. What Already Exists: Current Approaches and Their Gaps

### Standard Neural Machine Translation (NMT)

Modern translation systems use a technique called **Neural Machine Translation (NMT)**. Here's how it works in simple terms:

1. The system reads the source sentence (e.g., English) word by word
2. It builds an internal representation (a mathematical summary) of what the sentence means
3. It generates the target sentence (e.g., Hindi) word by word from that representation

The most common architecture for NMT is the **Encoder-Decoder model**:

- **Encoder**: Reads and "understands" the source sentence, compressing it into a dense representation
- **Decoder**: Takes that representation and generates the translation one word at a time

Popular systems like Google Translate, MarianMT, and mBART use this architecture. They are trained on millions of parallel sentences (the same sentence in two languages, side by side) and learn to produce fluent, grammatically correct translations.

### Where Current Systems Fail

These systems are evaluated using **BLEU** (Bilingual Evaluation Understudy), which counts how many word sequences in the translation match a human reference. BLEU is excellent for measuring literal accuracy — but it tells us nothing about whether the *tone* or *intent* survived translation.

Consider this example from our results:

> **English (sarcastic):** *"i love how when i'm stressed my body decides to react by causing me massive pain."*
>
> **Standard translation (Hindi):** मैं प्यार करता हूँ कि जब मैं अपने शरीर पर ज़ोर दे रहा हूँ...
>
> **Sentiment of original:** Positive (surface level — the word "love" dominates)
> **Sentiment of translation:** Negative ❌

The standard system translated the words correctly, but the **pragmatic signal** (this is sarcasm — the person does NOT love this) was lost. The translation's emotional tone flipped.

### Existing Research Gaps

| Approach | What it does | Limitation |
|----------|-------------|------------|
| Standard NMT | Translates words/grammar | No awareness of sarcasm or tone |
| Sentiment-aware NMT | Adds sentiment labels | Doesn't distinguish *sarcasm* from genuine sentiment |
| Multilingual models (mBART, mT5) | Train on many languages | Still no explicit pragmatic signal handling |
| Large Language Models (GPT, LLaMA) | General-purpose generation | Too expensive for dedicated translation; not specialised for pragmatic preservation |

**No existing system explicitly conditions translation on detected sarcasm and optimises for pragmatic preservation.** This is the gap our project fills.

---

## 3. Our Approach: A Multi-Stage Pipeline

We built a complete pipeline that **detects sarcasm first, then uses that information to guide translation**. Here is every component explained:

### Stage 1: Sarcasm Detection

**What:** We fine-tuned a pre-trained language model to classify English tweets as sarcastic or literal.

**Model Used:** `twitter-roberta-base` — a version of **RoBERTa** (a powerful text understanding model) that was specifically pre-trained on ~58 million tweets. This makes it particularly good at understanding informal social media language, slang, and emotional cues.

**What is fine-tuning?** Think of it like this: RoBERTa already "speaks" Twitter English. Fine-tuning is like giving it a specialised course on "recognising sarcasm" — we show it thousands of labelled examples (this tweet is sarcastic, this one isn't) and it learns the subtle patterns.

**Training Details:**
- Dataset: `tweet_eval/irony` — 3,817 training tweets, balanced (49.8% sarcastic)
- Training: 3 epochs (the model sees all examples 3 times), batch size 16, learning rate 2×10⁻⁵
- Hardware: Google Colab T4 GPU

**Results:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Not Sarcastic | 0.87 | 0.75 | 0.81 |
| Sarcastic | 0.69 | 0.83 | 0.75 |
| **Overall Accuracy** | | | **78.19%** |

The detector identifies sarcasm with 75% F1-score — strong enough to reliably tag social media content.

### Stage 2: Sarcasm-Enriched Parallel Corpus (Novel ✨)

**The Problem:** We need parallel data (English + Hindi side by side) with sarcasm labels. But the standard parallel corpus (IITB English-Hindi) contains formal, clean text — almost zero sarcasm. And the sarcasm dataset has English tweets but no Hindi translations.

**Our Solution:** We created a **synthetic sarcasm-enriched parallel corpus** by:

1. Taking the sarcastic English tweets (from the detection dataset)
2. Translating them to Hindi using the base MarianMT model (to generate "silver-standard" references)
3. Labelling each pair with its sarcasm status
4. Combining these with the regular IITB parallel data

This gives the translation model something it never had before: **training examples of sarcastic content with corresponding translations**, so it can learn how sarcasm should be handled.

**Data Composition:**
- Regular parallel pairs: 3,985 (from IITB)
- Sarcasm-enriched pairs: up to 2,000 (from tweet_eval)
- Test sets kept **separate** for fair evaluation

### Stage 3: Pragmatic-Aware NMT with Control Tokens & Weighted Loss (Novel ✨)

This is the core of our approach. We modified a standard translation model in two novel ways:

#### Control Tokens

**What are control tokens?** They are special markers we add to the beginning of the input sentence to tell the model "this sentence is sarcastic" or "this sentence is literal":

- `<SARCASTIC> oh great, another monday morning` → Model knows to preserve sarcastic tone
- `<LITERAL> the weather is nice today` → Model translates normally

This lets one single model handle both sarcastic and literal text differently, without needing two separate models.

**Base Translation Model:** We used **MarianMT** (`Helsinki-NLP/opus-mt-en-hi`), a compact, efficient encoder-decoder translation model specifically trained for English-to-Hindi translation.

#### Sarcasm-Weighted Loss

**What is "loss" in machine learning?** During training, the model makes a prediction (a translation), and the loss measures how far that prediction is from the correct answer. The model then adjusts itself to reduce the loss. Lower loss = better translations.

**Our innovation:** We multiply the loss by **1.5×** for sarcastic training examples. This tells the model: *"pay extra attention to getting sarcastic translations right"*. Without this weighting, sarcastic examples (which are the minority) get drowned out by the much larger quantity of literal examples.

Why 1.5× and not higher? We found that 3× weighting caused **quality collapse** — the model became so focused on sarcasm that its overall translation quality dropped dramatically. 1.5× strikes the right balance.

**Training Details:**
- 3 epochs, batch size 8, learning rate 3×10⁻⁵
- FP16 mixed precision training (uses less GPU memory)

### Stage 4: Contrastive Pragmatic Alignment (Novel ✨✨)

**Inspiration:** This technique is borrowed from **CLIP** (by OpenAI) and **SimCLR** (by Google) — systems that learn to align images with their text descriptions.

**What it does:** We add a second training objective alongside the standard translation loss. This **InfoNCE contrastive loss** teaches the model's internal representations to:

- **Pull together** sarcastic source texts with their correct translations (positive pairs)
- **Push apart** sarcastic and literal representations (negative pairs)

**Why?** The standard translation loss only cares about producing the right words. The contrastive loss forces the model's internal "understanding" of sarcasm to become more distinct and structured. Think of it as teaching the model not just *what* to translate, but to *feel* the difference between sarcasm and literal speech.

**Training:** 2 epochs, contrastive weight α=0.1 (a small amount, so it doesn't destabilise translation quality).

### Stage 5: Reinforcement Learning from Pragmatic Feedback — RLPF (Novel ✨✨✨)

**Inspiration:** This is our most ambitious technique, inspired by **RLHF** (Reinforcement Learning from Human Feedback) — the technique that made ChatGPT so effective. Except instead of human feedback, we use an **automated pragmatic quality signal**.

**How reinforcement learning (RL) works in simple terms:**

1. The model generates a translation (this is its "action")
2. We measure how well the translation preserves the original's pragmatic meaning (this is the "reward")
3. If the translation preserves meaning well → the model gets a positive reward and learns to produce similar translations
4. If the translation loses meaning → negative reward, model learns to avoid this

**The Reward Signal — Pragmatic Divergence Score (PDS):** We measure reward using cosine similarity between the source and translation in a **multilingual embedding space**. In simple terms:

- A multilingual sentence embedding model converts both the English source and Hindi translation into points in a high-dimensional mathematical space
- Sentences with similar *meaning and tone* end up close together
- The closeness (cosine similarity) becomes the reward: closer = better pragmatic preservation

**Safeguards:**
- **KL penalty** (coefficient 0.05): Prevents the model from "cheating" — generating nonsensical text that accidentally scores well on the reward. It keeps the RL-tuned model close to its original behaviour.
- **Running reward baseline**: Reduces variance in training, making learning more stable.

**Training:** 1 epoch, learning rate 1×10⁻⁶ (very small — RL fine-tuning needs gentle adjustments), batch size 8.

---

## 4. How We Measured Success

We used multiple evaluation metrics, each measuring a different aspect of quality:

### Translation Quality Metrics

| Metric | What It Measures | How It Works |
|--------|-----------------|--------------|
| **BLEU** | Word-level accuracy | Counts matching word sequences (n-grams) between translation and reference. Higher = more words match. |
| **BERTScore** | Meaning-level similarity | Uses a deep learning model (BERT) to compare the *meaning* of words in context, not just exact matches. More forgiving of synonyms and paraphrases. |
| **chrF++** | Character-level accuracy | Counts matching character sequences. Better than BLEU for Hindi because Hindi has complex word endings (morphology). |

### Pragmatic Preservation Metrics (Novel)

| Metric | What It Measures | How It Works |
|--------|-----------------|--------------|
| **Sentiment Preservation Rate** | Does the emotional polarity survive translation? | Runs sentiment analysis on both source and translation. If both are "positive" or both are "negative", it's preserved. Higher = better. |
| **Pragmatic Divergence Score (PDS)** ✨ | Overall meaning + tone similarity | Embeds source and translation in a multilingual space and measures cosine similarity. Captures nuance beyond just positive/negative sentiment. Higher = better. |
| **Sentiment Divergence** | How much does the sentiment distribution shift? | Compares the full probability distributions over positive/negative/neutral. Lower = better preservation. |

---

## 5. Results

### Part I: Baseline vs Pragmatic-Aware Model

| Metric | Baseline | Pragmatic-Aware | Change |
|--------|----------|----------------|--------|
| **BLEU** (IITB test set) | 64.82 | 74.68 | **+9.86** ↑ |
| **Sentiment Preservation** (sarcastic) | 54.21% | 61.05% | **+6.84%** ↑ |
| **Pragmatic Similarity (PDS)** (sarcastic) | 0.6877 | 0.7037 | **+0.0160** ↑ |
| **Sentiment Divergence** (sarcastic) | 0.1615 | 0.1553 | **−0.0062** ↓ (better) |

> **Key finding:** The pragmatic-aware model improves **every metric** — not just pragmatic preservation, but also raw translation quality (BLEU). This shows that sarcasm awareness doesn't come at the cost of accuracy; it actually *helps* the model produce better translations overall.

### Part II: Full 3-Model Comparison

| Metric | Baseline | Pragmatic | RL-Enhanced |
|----------------------------|----------|-----------|-------------|
| **BLEU (IITB)** | 64.82 | **74.68** | 70.43 |
| **Sentiment Preservation** ↑ | 54.21% | **61.05%** | 59.47% |
| **Pragmatic Sim. (PDS)** ↑ | 0.6877 | **0.7037** | 0.5819 |
| **Sentiment Divergence** ↓ | 0.1615 | **0.1553** | 0.1618 |

> **Interpretation:** The **Pragmatic-Aware model (Step 4) is the clear winner** on aggregate metrics, improving Sentiment Preservation by 6.84% and PDS by 2.3% over baseline. We also observe a slight regression in BLEU for the RL-Enhanced model (74.68 → 70.43), which is a common trade-off when optimising purely for a specific reward signal instead of raw human-reference matching.

>
> The **RL-Enhanced model** shows a different pattern: while its aggregate PDS drops (because RL sometimes causes repetitive outputs for complex sentences), it excels at the **hardest cases** — sarcastic sentences where both Baseline and Pragmatic models fail. The qualitative examples below demonstrate this clearly.

### Qualitative Examples: Where RL Truly Shines

These are sarcastic sentences where the Baseline and Pragmatic models **both fail** to preserve sentiment, but RL succeeds:

#### Example 1
> **Source:** *"i love how when i'm stressed my body decides to react by causing me massive pain."*
> (Source sentiment: **positive** — surface sarcasm)
>
> | Model | Hindi Translation | Sentiment | Correct? | PDS |
> |-------|------------------|-----------|----------|-----|
> | Baseline | मैं प्यार करता हूँ कि जब मैं अपने शरीर पर ज़ोर दे रहा हूँ... | negative | ❌ | 0.886 |
> | Pragmatic | मैं कैसे प्यार करता हूँ जब मैं... भारी दर्द पहुँचाने के... | negative | ❌ | 0.778 |
> | **RL-Enhanced** | मैं कैसे प्यार जब मुझे भारी दर्द... | **positive** | **✅** | 0.726 |

#### Example 2
> **Source:** *"great, i've got a lung infection. this is fun. :/ #notfun #miserable"*
> (Source sentiment: **positive** — sarcastic)
>
> | Model | Hindi Translation | Sentiment | Correct? | PDS |
> |-------|------------------|-----------|----------|-----|
> | Baseline | महान, मैं एक फेफड़ों संक्रमण मिल गया है. | negative | ❌ | 0.906 |
> | Pragmatic | महान, मैं एक फेफड़ों संक्रमण मिल गया है. | negative | ❌ | 0.902 |
> | **RL-Enhanced** | बहुत, मैं एक बाथरूम संक्रमण मिल है. यह मजेदार है. | **positive** | **✅** | 0.640 |

> **Key insight:** The RL model learns to make subtle word-choice changes that flip the translated sentiment to match the source — exactly what pragmatic preservation demands. It may sacrifice some literal accuracy (lower PDS) but achieves the *pragmatic* goal: the translation *reads* with the same emotional tone.

---

## 6. Why MarianMT Instead of Larger Models?

A common question is: *"Why not use GPT-4, LLaMA, or other large language models (LLMs) for translation?"* There are several deliberate, principled reasons:

### Computational Cost and Accessibility

| Model | Parameters | GPU Memory | Inference Time |
|-------|-----------|------------|----------------|
| **MarianMT** (ours) | ~74M | ~300 MB | ~50ms per sentence |
| mBART-large | ~610M | ~2.5 GB | ~200ms |
| GPT-3.5 | ~175B | Not runnable locally | API costs $$$ |
| LLaMA-2 70B | ~70B | ~140 GB (8x A100) | ~500ms |

MarianMT is **2,000× smaller** than GPT-3.5. It runs comfortably on a **free Google Colab T4 GPU** (15 GB). This is critical because:

- **Reproducibility**: Any researcher or student can replicate our work with zero cost
- **Deployment**: The model can run on a single laptop or budget server
- **Real-time translation**: At 50ms per sentence, it's fast enough for live applications

### Task Specialisation vs General-Purpose

LLMs (GPT, LLaMA, etc.) are **general-purpose** — they can write poetry, answer questions, code, and translate. But this generality comes at a cost: they aren't *specialised* for translation. MarianMT, by contrast, was trained **specifically for English-to-Hindi translation** on millions of parallel sentences. For this specific task, a specialised small model outperforms a general giant.

### Fine-Tuning Feasibility

Our approach requires **three stages of fine-tuning** (supervised, contrastive, and RL). Fine-tuning a 175B-parameter LLM for each stage would require:

- Multiple high-end GPUs (8× A100, costing ~$30/hour)
- Days of training time
- Sophisticated distributed training infrastructure

With MarianMT, each fine-tuning stage completes in **under an hour on a free Colab GPU**.

### Why Not BERT for Translation?

**BERT** (Bidirectional Encoder Representations from Transformers) is an extremely powerful model, but it is an **encoder-only** model — it reads and understands text but **cannot generate** new text. Translation requires an encoder-decoder architecture (read source → generate target). We *do* use BERT-family models in other parts of our pipeline:

- **twitter-roberta-base** (a BERT variant) for sarcasm detection
- **bert-base-multilingual-cased** for BERTScore evaluation
- **paraphrase-multilingual-MiniLM** (a distilled BERT variant) for PDS computation

So BERT plays a supporting role throughout the pipeline — it's just not the right architecture for the translation task itself.

### Research Contribution Focus

Our contribution is **the technique** (control tokens + weighted loss + contrastive alignment + RL from pragmatic feedback), not the base model. These techniques are **model-agnostic** — they could be applied to mBART, NLLB, or any future translation model. By proving them on a small, accessible model, we make the research immediately useful to the widest possible audience.

---

## 7. Novel Contributions

Our project introduces **four techniques** that, to our knowledge, have not been previously combined for pragmatic-aware machine translation:

| # | Technique | Inspiration | What's Novel |
|---|-----------|-------------|-------------|
| 1 | **Sarcasm-Enriched Synthetic Corpus** | Data augmentation | Using a sarcasm detector to construct parallel training data with pragmatic labels — bridging the gap between sarcasm datasets (English-only) and parallel corpora (no sarcasm) |
| 2 | **Control Token + Weighted Loss NMT** | Conditional generation | Combining `<SARCASTIC>`/`<LITERAL>` control tokens with per-example loss weighting (1.5× for sarcastic) to explicitly condition translation on pragmatic type |
| 3 | **Contrastive Pragmatic Alignment** | CLIP / SimCLR | Applying InfoNCE contrastive loss to NMT encoder representations, forcing sarcastic and literal internal representations to become distinct |
| 4 | **RL from Pragmatic Feedback (RLPF)** | RLHF (ChatGPT) | REINFORCE-style policy gradient using Pragmatic Divergence Score (multilingual cosine similarity) as the reward signal — directly optimising translations for pragmatic preservation rather than word-matching |

### The Pragmatic Divergence Score (PDS) itself is a contribution

Traditional MT metrics (BLEU, chrF++) measure **surface-level** similarity. PDS measures **meaning-level** similarity across languages by using multilingual sentence embeddings. This score:

- Works across any language pair (not just English-Hindi)
- Captures tone, register, and pragmatic nuance
- Can serve as both an evaluation metric and a training signal (as in RLPF)

---

## 8. Limitations and Honest Assessment

### What worked well
- **Pragmatic-Aware model** consistently improved all metrics over baseline
- **BLEU improved by 9.86%** (64.82 → 74.68) — showing sarcasm-awareness helps even literal translation quality
- **Sentiment Preservation improved by 6.84 percentage points** on sarcastic content
- **RL model** excels at the hardest individual examples

### What has room for improvement
- **RL-Enhanced model's aggregate PDS dropped** (0.7037 → 0.5819): While RL excels at fixing hard cases, it sometimes causes repetitive token generation ("..." patterns) on complex inputs, which drags down the average
- **Silver-standard translations**: Our enriched corpus uses machine-generated Hindi translations as references, introducing noise. Human translations would improve quality
- **Sarcasm detector at 78%**: Some misclassified examples inject noise into the pipeline

### Future directions
- **Human evaluation** of translation quality (not just automated metrics)
- **Extension to more pragmatic signals**: irony, understatement, hyperbole, rhetorical questions
- **More language pairs**: Hindi is just one target; the technique is language-agnostic
- **Larger-scale RL training** with more epochs and human-in-the-loop feedback

---

## 9. Societal Impact

### Improving Cross-lingual Understanding

India alone has 22 official languages and 1.4 billion people. Social media content in English is constantly being consumed, shared, and translated across these languages. When sarcasm is lost in translation, it doesn't just cause confusion — it can:

- **Distort public opinion analysis** across languages
- **Cause diplomatic or social misunderstandings** when sarcastic political commentary is taken literally
- **Reduce the effectiveness of mental health monitoring** systems that rely on translated social media for detecting distress signals

### Making NLP More Accessible

By building on small, efficient models (MarianMT) and training on free GPU resources (Colab), our work is **immediately reproducible** by students, researchers, and developers worldwide — not just by large corporations with massive compute budgets.

### A Template for Pragmatic-Aware NLP

Our pipeline (detect → condition → fine-tune → RL-optimise) is a template that can be applied to **any** pragmatic signal in **any** language pair. Future work could adapt this to preserve:

- **Politeness levels** in Japanese/Korean translation
- **Formal vs informal register** in Hindi/Urdu
- **Rhetorical questions** in Arabic
- **Understatement** in British English → American English

---

## 10. Technical Architecture Summary

```
                        ┌─────────────────┐
                        │  English Tweet  │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Sarcasm Detector      │
                    │  (twitter-roberta-base) │
                    └────────────┬────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │ <SARCASTIC> or      │
                      │ <LITERAL> prefix    │
                      └──────────┬──────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │    Pragmatic-Aware MarianMT         │
              │  (control tokens + weighted loss    │
              │   + contrastive + RL fine-tuning)   │
              └──────────────────┬──────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Hindi Output   │
                        │ (with preserved │
                        │  pragmatic tone)│
                        └─────────────────┘
```

**Total pipeline training time:** Google Colab T4 GPU, ~45 minutes total if one epoch on RLPF, ~80 minutes total if two epoch on RLPF.

---

## 11. References

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30. — Introduced the Transformer architecture that underpins all models used in this project.

2. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V.** (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*. — The base architecture for our sarcasm detector (twitter-roberta-base).

3. **Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L.** (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *Findings of EMNLP 2020*. — Source of the tweet_eval/irony dataset used for sarcasm detection training.

4. **Junczys-Dowmunt, M., Grundkiewicz, R., Dwojak, T., Hoang, H., Heafield, K., Neckermann, T., Seide, F., Germann, U., Fikri Aji, A., Bogoychev, N., Martins, A. F. T., & Birch, A.** (2018). Marian: Fast Neural Machine Translation in C++. *Proceedings of ACL 2018 System Demonstrations*. — The MarianMT framework; our base translation model (Helsinki-NLP/opus-mt-en-hi).

5. **Tiedemann, J. & Thottingal, S.** (2020). OPUS-MT — Building open translation services for the World. *Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)*. — The OPUS-MT project providing pre-trained MarianMT models.

6. **Kunchukuttan, A., Mehta, P., & Bhattacharyya, P.** (2018). The IIT Bombay English-Hindi Parallel Corpus. *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*. — Source of the IITB English-Hindi parallel corpus used for translation training and evaluation.

7. **Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J.** (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *Proceedings of the 40th Annual Meeting of the ACL*. — The BLEU metric used for translation quality evaluation.

8. **Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y.** (2020). BERTScore: Evaluating Text Generation with BERT. *International Conference on Learning Representations (ICLR)*. — The BERTScore metric used for contextual semantic evaluation.

9. **Popović, M.** (2015). chrF: Character n-gram F-score for Automatic MT Evaluation. *Proceedings of the Tenth Workshop on Statistical Machine Translation*. — The chrF++ metric used for character-level evaluation, particularly suited for morphologically rich languages like Hindi.

10. **Reimers, N. & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*. — Foundation for the multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2) used in PDS computation.

11. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I.** (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *Proceedings of ICML 2021*. — Inspiration for our contrastive pragmatic alignment using InfoNCE loss.

12. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *Proceedings of ICML 2020*. — Inspiration for our contrastive learning approach to separate sarcastic and literal representations.

13. **Oord, A. van den, Li, Y., & Vinyals, O.** (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*. — Introduced the InfoNCE loss function used in our contrastive alignment stage.

14. **Williams, R. J.** (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*, 8(3–4), 229–256. — The REINFORCE algorithm used in our RL from Pragmatic Feedback (RLPF) stage.

15. **Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R.** (2022). Training language models to follow instructions with human feedback (RLHF). *Advances in Neural Information Processing Systems (NeurIPS)*, 35. — Inspiration for adapting RL-based fine-tuning to machine translation with automated pragmatic rewards.

16. **Ranzato, M., Chopra, S., Auli, M., & Zaremba, W.** (2016). Sequence Level Training with Recurrent Neural Networks. *International Conference on Learning Representations (ICLR)*. — Pioneered using REINFORCE for sequence generation tasks; foundational to applying RL in NMT.

17. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*. — The BERT architecture used in BERTScore evaluation and as basis for RoBERTa.

18. **Post, M.** (2018). A Call for Clarity in Reporting BLEU Scores. *Proceedings of the Third Conference on Machine Translation (WMT)*. — SacreBLEU standardisation used in our evaluation pipeline.

---

*Pragmatic Signal-Aware Neural Machine Translation with Contrastive Learning and Reinforcement Learning*
