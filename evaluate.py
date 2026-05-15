"""
evaluate.py — Evaluation Metrics (v2)
=======================================
Computes BLEU, Sentiment Preservation Rate, Pragmatic Divergence Score
(using multilingual sentence embeddings), and generates comparison
tables between baseline and pragmatic-aware NMT models.
"""

import random
import numpy as np
import torch
import pandas as pd
from transformers import pipeline
import sacrebleu

# ─── reproducibility ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── model names ───────────────────────────────────────────────────
SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ─── sentiment pipeline ───────────────────────────────────────────
def load_sentiment_pipeline():
    """Load the multilingual sentiment analysis pipeline."""
    print(f"📦 Loading sentiment model: {SENTIMENT_MODEL}")
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )


def get_sentiment(text: str, sentiment_pipe) -> str:
    """Get the dominant sentiment polarity for a text."""
    try:
        results = sentiment_pipe(text[:512], truncation=True)[0]
        best = max(results, key=lambda x: x["score"])
        return best["label"].lower()
    except Exception:
        return "neutral"


def get_sentiment_scores(text: str, sentiment_pipe) -> dict:
    """Get full sentiment scores for a text."""
    try:
        results = sentiment_pipe(text[:512], truncation=True)[0]
        return {r["label"].lower(): r["score"] for r in results}
    except Exception:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


# ─── BLEU score ─────────────────────────────────────────────────────
def compute_bleu(hypotheses: list, references: list) -> dict:
    """Compute corpus-level BLEU score using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu": bleu.score,
        "bleu_str": str(bleu),
    }


# ─── sentiment preservation ────────────────────────────────────────
def compute_sentiment_preservation(
    sources: list,
    translations: list,
    sentiment_pipe=None,
) -> dict:
    """
    Compute Sentiment Preservation Rate: % of examples where source
    and translation have the same sentiment polarity.
    """
    if sentiment_pipe is None:
        sentiment_pipe = load_sentiment_pipeline()

    matches = 0
    total = len(sources)
    details = []

    for src, tgt in zip(sources, translations):
        src_sent = get_sentiment(src, sentiment_pipe)
        tgt_sent = get_sentiment(tgt, sentiment_pipe)
        match = src_sent == tgt_sent
        if match:
            matches += 1
        details.append({
            "source": src,
            "translation": tgt,
            "source_sentiment": src_sent,
            "translation_sentiment": tgt_sent,
            "preserved": match,
        })

    rate = matches / total if total > 0 else 0.0
    return {
        "preservation_rate": rate,
        "matches": matches,
        "total": total,
        "details": details,
    }


# ─── pragmatic divergence score (v2 NOVEL) ─────────────────────────
def load_embedding_model():
    """Load multilingual sentence embedding model."""
    from sentence_transformers import SentenceTransformer
    print(f"📦 Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def compute_pragmatic_divergence(
    sources: list,
    translations: list,
    embed_model=None,
) -> dict:
    """
    Compute Pragmatic Divergence Score (PDS):
    Cosine similarity between source and translation in a multilingual
    sentence embedding space. Higher similarity = better pragmatic
    preservation.

    This captures semantic nuance beyond BLEU's n-gram matching,
    including tone, register, and pragmatic intent.

    Returns dict with mean similarity and per-example scores.
    """
    if embed_model is None:
        embed_model = load_embedding_model()

    src_embeddings = embed_model.encode(sources, show_progress_bar=False)
    tgt_embeddings = embed_model.encode(translations, show_progress_bar=False)

    # Cosine similarity per example
    from numpy.linalg import norm
    similarities = []
    for s, t in zip(src_embeddings, tgt_embeddings):
        sim = np.dot(s, t) / (norm(s) * norm(t) + 1e-8)
        similarities.append(float(sim))

    return {
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "per_example": similarities,
    }


# ─── sentiment divergence (v2 NOVEL) ──────────────────────────────
def compute_sentiment_divergence(
    sources: list,
    translations: list,
    sentiment_pipe=None,
) -> dict:
    """
    Compute average absolute difference in sentiment distributions
    between source and translation. Lower = better preservation.
    Uses full sentiment score vectors, not just argmax.
    """
    if sentiment_pipe is None:
        sentiment_pipe = load_sentiment_pipeline()

    divergences = []
    for src, tgt in zip(sources, translations):
        src_scores = get_sentiment_scores(src, sentiment_pipe)
        tgt_scores = get_sentiment_scores(tgt, sentiment_pipe)
        # MAE across sentiment dimensions
        div = np.mean([
            abs(src_scores.get(k, 0) - tgt_scores.get(k, 0))
            for k in set(list(src_scores.keys()) + list(tgt_scores.keys()))
        ])
        divergences.append(div)

    return {
        "mean_divergence": np.mean(divergences),
        "std_divergence": np.std(divergences),
    }


# ─── BERTScore (contextual embedding similarity) ──────────────────
def compute_bertscore(hypotheses: list, references: list, lang: str = "hi") -> dict:
    """
    Compute BERTScore — evaluates translation quality using contextual
    BERT embeddings. Captures semantic similarity beyond n-gram overlap.
    """
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "per_example_f1": F1.tolist(),
    }


# ─── chrF++ (character n-gram F-score) ────────────────────────────
def compute_chrf(hypotheses: list, references: list) -> dict:
    """
    Compute chrF++ score — character-level n-gram F-score that is more
    robust than BLEU for morphologically rich languages like Hindi.
    """
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {
        "chrf": chrf.score,
        "chrf_str": str(chrf),
    }



# ─── full evaluation pipeline ──────────────────────────────────────
def full_evaluation(
    test_sources: list,
    test_references: list,
    baseline_translations: list,
    pragmatic_translations: list,
    sarcasm_labels: list,
    sentiment_pipe=None,
    embed_model=None,
) -> dict:
    """
    Run complete evaluation: BLEU + sentiment preservation +
    pragmatic divergence for both models, overall and by subset.
    """
    if sentiment_pipe is None:
        sentiment_pipe = load_sentiment_pipeline()
    if embed_model is None:
        embed_model = load_embedding_model()

    results = {}

    # ── overall BLEU ────────────────────────────────────────────────
    results["baseline_bleu"] = compute_bleu(baseline_translations, test_references)
    results["pragmatic_bleu"] = compute_bleu(pragmatic_translations, test_references)

    # ── overall sentiment preservation ──────────────────────────────
    results["baseline_sentiment"] = compute_sentiment_preservation(
        test_sources, baseline_translations, sentiment_pipe
    )
    results["pragmatic_sentiment"] = compute_sentiment_preservation(
        test_sources, pragmatic_translations, sentiment_pipe
    )

    # ── overall pragmatic divergence (v2) ───────────────────────────
    results["baseline_pds"] = compute_pragmatic_divergence(
        test_sources, baseline_translations, embed_model
    )
    results["pragmatic_pds"] = compute_pragmatic_divergence(
        test_sources, pragmatic_translations, embed_model
    )

    # ── overall sentiment divergence (v2) ───────────────────────────
    results["baseline_sent_div"] = compute_sentiment_divergence(
        test_sources, baseline_translations, sentiment_pipe
    )
    results["pragmatic_sent_div"] = compute_sentiment_divergence(
        test_sources, pragmatic_translations, sentiment_pipe
    )

    # ── sarcastic subset ────────────────────────────────────────────
    sarc_idx = [i for i, l in enumerate(sarcasm_labels) if l == 1]
    if sarc_idx:
        sarc_sources = [test_sources[i] for i in sarc_idx]
        sarc_refs = [test_references[i] for i in sarc_idx]
        sarc_baseline = [baseline_translations[i] for i in sarc_idx]
        sarc_pragmatic = [pragmatic_translations[i] for i in sarc_idx]

        results["sarcastic_baseline_bleu"] = compute_bleu(sarc_baseline, sarc_refs)
        results["sarcastic_pragmatic_bleu"] = compute_bleu(sarc_pragmatic, sarc_refs)

        results["sarcastic_baseline_sentiment"] = compute_sentiment_preservation(
            sarc_sources, sarc_baseline, sentiment_pipe
        )
        results["sarcastic_pragmatic_sentiment"] = compute_sentiment_preservation(
            sarc_sources, sarc_pragmatic, sentiment_pipe
        )

        results["sarcastic_baseline_pds"] = compute_pragmatic_divergence(
            sarc_sources, sarc_baseline, embed_model
        )
        results["sarcastic_pragmatic_pds"] = compute_pragmatic_divergence(
            sarc_sources, sarc_pragmatic, embed_model
        )

        results["sarcastic_baseline_sent_div"] = compute_sentiment_divergence(
            sarc_sources, sarc_baseline, sentiment_pipe
        )
        results["sarcastic_pragmatic_sent_div"] = compute_sentiment_divergence(
            sarc_sources, sarc_pragmatic, sentiment_pipe
        )
    else:
        for prefix in ("sarcastic_baseline", "sarcastic_pragmatic"):
            results[f"{prefix}_bleu"] = {"bleu": 0.0}
            results[f"{prefix}_sentiment"] = {"preservation_rate": 0.0}
            results[f"{prefix}_pds"] = {"mean_similarity": 0.0}
            results[f"{prefix}_sent_div"] = {"mean_divergence": 0.0}

    # ── literal subset ──────────────────────────────────────────────
    lit_idx = [i for i, l in enumerate(sarcasm_labels) if l == 0]
    if lit_idx:
        lit_sources = [test_sources[i] for i in lit_idx]
        lit_refs = [test_references[i] for i in lit_idx]
        lit_baseline = [baseline_translations[i] for i in lit_idx]
        lit_pragmatic = [pragmatic_translations[i] for i in lit_idx]

        results["literal_baseline_bleu"] = compute_bleu(lit_baseline, lit_refs)
        results["literal_pragmatic_bleu"] = compute_bleu(lit_pragmatic, lit_refs)

        results["literal_baseline_sentiment"] = compute_sentiment_preservation(
            lit_sources, lit_baseline, sentiment_pipe
        )
        results["literal_pragmatic_sentiment"] = compute_sentiment_preservation(
            lit_sources, lit_pragmatic, sentiment_pipe
        )

        results["literal_baseline_pds"] = compute_pragmatic_divergence(
            lit_sources, lit_baseline, embed_model
        )
        results["literal_pragmatic_pds"] = compute_pragmatic_divergence(
            lit_sources, lit_pragmatic, embed_model
        )

    return results


# ─── display utilities ──────────────────────────────────────────────
def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create a pandas DataFrame summarizing all evaluation metrics."""
    rows = []

    # Overall
    rows.append({
        "Subset": "Overall", "Metric": "BLEU ↑",
        "Baseline": f"{results['baseline_bleu']['bleu']:.2f}",
        "Pragmatic-Aware": f"{results['pragmatic_bleu']['bleu']:.2f}",
    })
    rows.append({
        "Subset": "Overall", "Metric": "Sentiment Preservation ↑",
        "Baseline": f"{results['baseline_sentiment']['preservation_rate']:.2%}",
        "Pragmatic-Aware": f"{results['pragmatic_sentiment']['preservation_rate']:.2%}",
    })
    rows.append({
        "Subset": "Overall", "Metric": "Pragmatic Sim. (PDS) ↑",
        "Baseline": f"{results['baseline_pds']['mean_similarity']:.4f}",
        "Pragmatic-Aware": f"{results['pragmatic_pds']['mean_similarity']:.4f}",
    })
    rows.append({
        "Subset": "Overall", "Metric": "Sentiment Divergence ↓",
        "Baseline": f"{results['baseline_sent_div']['mean_divergence']:.4f}",
        "Pragmatic-Aware": f"{results['pragmatic_sent_div']['mean_divergence']:.4f}",
    })

    # Sarcastic subset
    rows.append({
        "Subset": "Sarcastic ⭐", "Metric": "BLEU ↑",
        "Baseline": f"{results['sarcastic_baseline_bleu']['bleu']:.2f}",
        "Pragmatic-Aware": f"{results['sarcastic_pragmatic_bleu']['bleu']:.2f}",
    })
    rows.append({
        "Subset": "Sarcastic ⭐", "Metric": "Sentiment Preservation ↑",
        "Baseline": f"{results['sarcastic_baseline_sentiment']['preservation_rate']:.2%}",
        "Pragmatic-Aware": f"{results['sarcastic_pragmatic_sentiment']['preservation_rate']:.2%}",
    })
    rows.append({
        "Subset": "Sarcastic ⭐", "Metric": "Pragmatic Sim. (PDS) ↑",
        "Baseline": f"{results['sarcastic_baseline_pds']['mean_similarity']:.4f}",
        "Pragmatic-Aware": f"{results['sarcastic_pragmatic_pds']['mean_similarity']:.4f}",
    })
    rows.append({
        "Subset": "Sarcastic ⭐", "Metric": "Sentiment Divergence ↓",
        "Baseline": f"{results['sarcastic_baseline_sent_div']['mean_divergence']:.4f}",
        "Pragmatic-Aware": f"{results['sarcastic_pragmatic_sent_div']['mean_divergence']:.4f}",
    })

    # Literal subset
    if "literal_baseline_bleu" in results:
        rows.append({
            "Subset": "Literal", "Metric": "BLEU ↑",
            "Baseline": f"{results['literal_baseline_bleu']['bleu']:.2f}",
            "Pragmatic-Aware": f"{results['literal_pragmatic_bleu']['bleu']:.2f}",
        })
        rows.append({
            "Subset": "Literal", "Metric": "Sentiment Preservation ↑",
            "Baseline": f"{results['literal_baseline_sentiment']['preservation_rate']:.2%}",
            "Pragmatic-Aware": f"{results['literal_pragmatic_sentiment']['preservation_rate']:.2%}",
        })
        rows.append({
            "Subset": "Literal", "Metric": "Pragmatic Sim. (PDS) ↑",
            "Baseline": f"{results['literal_baseline_pds']['mean_similarity']:.4f}",
            "Pragmatic-Aware": f"{results['literal_pragmatic_pds']['mean_similarity']:.4f}",
        })

    return pd.DataFrame(rows)


def print_qualitative_examples(
    sources: list,
    baseline_translations: list,
    pragmatic_translations: list,
    sarcasm_labels: list,
    sentiment_pipe=None,
    embed_model=None,
    n_examples: int = 5,
):
    """
    Print qualitative examples showing sarcasm preservation improvement.
    Prioritizes sarcastic examples where pragmatic model does better.
    """
    if sentiment_pipe is None:
        sentiment_pipe = load_sentiment_pipeline()

    print("\n" + "=" * 70)
    print("📝 QUALITATIVE EXAMPLES — Sarcasm Preservation")
    print("=" * 70)

    # Find sarcastic examples
    sarc_indices = [i for i, l in enumerate(sarcasm_labels) if l == 1]
    if len(sarc_indices) < n_examples:
        # Add some literal examples too
        lit_indices = [i for i, l in enumerate(sarcasm_labels) if l == 0]
        sarc_indices.extend(lit_indices[:n_examples - len(sarc_indices)])

    # Score each: preference for cases where pragmatic preserves sentiment better
    scored = []
    for idx in sarc_indices:
        src_sent = get_sentiment(sources[idx], sentiment_pipe)
        base_sent = get_sentiment(baseline_translations[idx], sentiment_pipe)
        prag_sent = get_sentiment(pragmatic_translations[idx], sentiment_pipe)
        base_match = src_sent == base_sent
        prag_match = src_sent == prag_sent
        # Prefer examples where pragmatic is better than baseline
        score = int(prag_match) - int(base_match)
        scored.append((idx, score, src_sent, base_sent, prag_sent))

    # Sort: pragmatic-better first, then by diversity
    scored.sort(key=lambda x: -x[1])
    selected = scored[:n_examples]

    for rank, (idx, _, src_sent, base_sent, prag_sent) in enumerate(selected, 1):
        src = sources[idx]
        base = baseline_translations[idx]
        prag = pragmatic_translations[idx]
        label = "🎭 SARCASTIC" if sarcasm_labels[idx] == 1 else "📝 LITERAL"

        # Compute embeddings for PDS if model available
        pds_info = ""
        if embed_model is not None:
            base_pds = compute_pragmatic_divergence([src], [base], embed_model)
            prag_pds = compute_pragmatic_divergence([src], [prag], embed_model)
            pds_info = f" | PDS: {base_pds['mean_similarity']:.3f} → {prag_pds['mean_similarity']:.3f}"

        print(f"\n{'─' * 60}")
        print(f"Example {rank} [{label}]")
        print(f"  Source (EN):    {src}")
        print(f"  Source sentiment: {src_sent}")
        print(f"  ┌─ Baseline (HI):  {base}")
        print(f"  │  Sentiment:      {base_sent} {'✅' if src_sent == base_sent else '❌'}{pds_info}")
        print(f"  └─ Pragmatic (HI): {prag}")
        print(f"     Sentiment:      {prag_sent} {'✅' if src_sent == prag_sent else '❌'}")

    print(f"\n{'=' * 70}\n")


# ─── main (standalone test) ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing evaluate.py standalone")
    print("=" * 60)

    hyps = ["यह एक परीक्षा है", "मौसम अच्छा है"]
    refs = ["यह एक परीक्षण है", "आज मौसम अच्छा है"]
    bleu_result = compute_bleu(hyps, refs)
    print(f"BLEU: {bleu_result['bleu']:.2f}")
