from __future__ import annotations

# Cross-encoder re-ranker
# ─────────────────────────────────────────────────────────────────────────────
# Re-ranking is a two-stage retrieval improvement:
#
#   Stage 1 (bi-encoder, fast): embed query + all docs separately → cosine
#           similarity → retrieve top-N candidates (e.g. 20)
#
#   Stage 2 (cross-encoder, accurate): score each (query, candidate) pair
#           through a full transformer — much more accurate relevance signal,
#           but only runs on the small candidate set so latency stays low.
#
# Recommended models (all free, local):
#   cross-encoder/ms-marco-MiniLM-L-6-v2   ~22MB  fast,  good quality
#   cross-encoder/ms-marco-MiniLM-L-12-v2  ~66MB  slow,  better quality
#   BAAI/bge-reranker-base                  ~110MB medium, excellent quality
#
# The model is lazy-loaded and cached for the lifetime of the process.

_MODEL_CACHE: dict[str, object] = {}


def _load_cross_encoder(model_name: str):
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Re-ranking requires sentence-transformers. "
                "Run: pip install sentence-transformers"
            ) from exc
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def rerank(
    query: str,
    candidates: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 4,
) -> list[dict]:
    """Re-score `candidates` using a cross-encoder and return the top `top_k`.

    Each candidate must have a "text" field. The returned list is sorted by
    rerank_score descending and contains the same fields as the input plus:
      - rerank_score:   float — raw cross-encoder logit (higher = more relevant)
      - biencoder_score: float — the original bi-encoder cosine score (preserved)

    Falls back to the original bi-encoder ordering on any error.
    """
    if not candidates:
        return candidates

    try:
        model = _load_cross_encoder(model_name)
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)  # numpy array of floats

        reranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["biencoder_score"] = item.get("score", 0.0)
            item["rerank_score"] = float(score)
            item["score"] = float(score)   # overwrite "score" so downstream code
            reranked.append(item)          # sees the better signal consistently

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    except Exception as exc:
        # Never crash the pipeline — degrade gracefully to bi-encoder order
        print(f"[reranker] Re-ranking failed ({exc}). Using bi-encoder order.")
        return candidates[:top_k]
