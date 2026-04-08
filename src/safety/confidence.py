from __future__ import annotations

import re

# ── NLI model (lazy-loaded once per process) ──────────────────────────────────
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_NLI_LABEL_ENTAILMENT = 1   # label order: [contradiction, entailment, neutral]
_nli_model_cache = None


def _get_nli_model():
    global _nli_model_cache
    if _nli_model_cache is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        _nli_model_cache = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model_cache


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip() and len(s.split()) >= 4]


# ── Retrieval confidence ──────────────────────────────────────────────────────

def compute_confidence(results: list[dict]) -> float:
    """Normalise top retrieval dot-product score to [0, 1]."""
    if not results:
        return 0.0
    top_score = max(float(r.get("score", 0.0)) for r in results)
    return max(0.0, min(1.0, (top_score + 1.0) / 2.0))


# ── Type 1 — Retrieval failure ────────────────────────────────────────────────

def detect_type1(confidence: float, threshold: float = 0.35) -> dict:
    """
    Type 1 hallucination: the retrieved chunks are not relevant to the query.
    Measured by retrieval confidence (normalised dot-product score).

    When confidence is low, the LLM has no good grounding to answer from
    and is more likely to generate from memory (hallucinate).

    Returns a structured dict — not just a bool.
    """
    flagged = confidence < threshold
    if confidence >= 0.75:
        risk_level = "low"
        reason = "Retrieval found highly relevant context."
    elif confidence >= threshold:
        risk_level = "medium"
        reason = "Retrieval found moderately relevant context — answer may be partially grounded."
    else:
        risk_level = "high"
        reason = (
            f"Retrieval confidence {confidence:.2f} is below threshold {threshold:.2f}. "
            "The retrieved chunks may not be relevant — LLM may answer from memory."
        )
    return {
        "flagged": flagged,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "risk_level": risk_level,
        "reason": reason,
    }


# ── Type 2 — Generation unfaithfulness ───────────────────────────────────────

def detect_type2_nli(
    answer: str,
    source_texts: list[str],
    threshold: float = 0.5,
    max_sentences: int = 6,
) -> dict:
    """
    Type 2 hallucination: the answer makes claims not supported by the retrieved context.
    Uses a cross-encoder NLI model to check if each sentence in the answer
    is *entailed* by the retrieved context.

    Score = fraction of answer sentences that are entailed.
      1.0 → fully grounded in context (no hallucination)
      0.0 → no sentences supported by context (full hallucination)

    Requires sentence-transformers (already a project dependency).
    Falls back gracefully if the model cannot be loaded.
    """
    # Strip simple-generation prefix — it's not a claim
    cleaned = answer.strip()
    for prefix in ("Answer based on retrieved context:", "[Fallback reason:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].split("\n")[0].strip()

    if not cleaned or not source_texts:
        return {
            "flagged": False,
            "score": 1.0,
            "entailed": 0,
            "total_sentences": 0,
            "detail": [],
            "model": _NLI_MODEL_NAME,
            "skipped": True,
            "skip_reason": "Empty answer or no source context.",
        }

    try:
        model = _get_nli_model()
    except Exception as exc:
        return {
            "flagged": False,
            "score": 0.5,
            "entailed": 0,
            "total_sentences": 0,
            "detail": [],
            "model": _NLI_MODEL_NAME,
            "error": f"NLI model unavailable: {exc}",
        }

    sentences = _split_sentences(cleaned)
    if not sentences:
        return {
            "flagged": False,
            "score": 1.0,
            "entailed": 0,
            "total_sentences": 0,
            "detail": [],
            "model": _NLI_MODEL_NAME,
            "skipped": True,
            "skip_reason": "No scoreable sentences in answer.",
        }

    # Cap to first N sentences — avoids runaway compute on long LLM answers
    sentences = sentences[:max_sentences]

    context = " ".join(source_texts)
    pairs = [(context, s) for s in sentences]

    try:
        scores = model.predict(pairs)   # shape: (n_sentences, 3)
    except Exception as exc:
        return {
            "flagged": False,
            "score": 0.5,
            "entailed": 0,
            "total_sentences": len(sentences),
            "detail": [],
            "model": _NLI_MODEL_NAME,
            "error": f"NLI inference failed: {exc}",
        }

    _labels = ["contradiction", "entailment", "neutral"]
    detail = []
    entailed_count = 0

    for sentence, score_vec in zip(sentences, scores):
        label = _labels[int(score_vec.argmax())]
        entailment_prob = float(score_vec[_NLI_LABEL_ENTAILMENT])
        if label == "entailment":
            entailed_count += 1
        detail.append({
            "sentence": sentence,
            "label": label,
            "entailment_prob": round(entailment_prob, 4),
        })

    score = entailed_count / len(sentences)

    return {
        "flagged": score < threshold,
        "score": round(score, 4),
        "entailed": entailed_count,
        "total_sentences": len(sentences),
        "threshold": threshold,
        "detail": detail,
        "model": _NLI_MODEL_NAME,
    }


# ── Backward-compat shim ──────────────────────────────────────────────────────

def hallucination_flag(confidence: float, threshold: float = 0.35) -> bool:
    """Legacy helper — use detect_type1() for structured output."""
    return confidence < threshold
