from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    if not expected_doc_ids:
        return 0.0
    topk = set(retrieved_doc_ids[:k])
    expected = set(expected_doc_ids)
    hit = len(topk & expected)
    return hit / len(expected)


def mrr(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    expected = set(expected_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / i
    return 0.0


def faithfulness_proxy(answer: str, source_texts: list[str]) -> float:
    """Token overlap between answer and source texts.
    Fast but unreliable — counts shared words, not shared meaning.
    Inflated when generation provider is 'simple' (answer = concatenated chunks).
    """
    if not answer.strip() or not source_texts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    src_tokens = set(" ".join(source_texts).lower().split())
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & src_tokens)
    return overlap / len(answer_tokens)


def faithfulness_bertscore_batch(
    answers: list[str], contexts: list[str]
) -> tuple[list[float], bool, str | None]:
    """Compute BERTScore Precision for a batch of (answer, context) pairs.

    Returns:
        (scores, used_fallback, error_reason)
        - scores: per-item float list
        - used_fallback: True if BERTScore failed and token overlap was used instead
        - error_reason: human-readable reason for fallback, or None on success

    BERTScore Precision measures how much of the answer is semantically
    grounded in the retrieved context — using BERT token embeddings instead
    of raw word overlap. A score of 1.0 means every token in the answer has
    a near-identical match in the context.

    Falls back to faithfulness_proxy per item if bert-score is not installed.
    Install with: pip install bert-score
    """
    if not answers:
        return [], False, None
    try:
        from bert_score import score as bs_score  # type: ignore
        P, _R, _F1 = bs_score(
            answers,
            contexts,
            lang="en",
            model_type="distilbert-base-uncased",  # lightweight, no GPU needed
            verbose=False,
        )
        return [float(p) for p in P], False, None
    except ImportError:
        reason = "bert-score not installed (run: pip install bert-score)"
        print(f"[faithfulness_bertscore] {reason}. Falling back to token overlap.")
        return [faithfulness_proxy(a, [c]) for a, c in zip(answers, contexts)], True, reason
    except Exception as exc:
        reason = str(exc)
        print(f"[faithfulness_bertscore] Failed: {reason}. Falling back to token overlap.")
        return [faithfulness_proxy(a, [c]) for a, c in zip(answers, contexts)], True, reason
