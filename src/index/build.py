from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from src.chunking.chunker import chunk_documents
from src.embeddings.providers import build_embedder
from src.index.vector_index import VectorIndex
from src.ingestion.loader import load_documents, persist_raw_docs


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embed_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_embed_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=True), encoding="utf-8")


def build_index(config: dict, progress_callback: Callable[[str], None] | None = None) -> dict:
    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    emit("Loading documents")
    data_config = config.get("data", {})
    docs = load_documents(config["paths"]["data_dir"], data_config=data_config)
    persist_raw_docs(docs, config["paths"]["raw_docs_dir"])

    chunking_cfg = config.get("chunking", {})
    emit(f"Chunking documents ({chunking_cfg.get('strategy', 'fixed')})")
    chunks = chunk_documents(
        docs,
        chunk_size_words=int(chunking_cfg.get("chunk_size_words", 180)),
        chunk_overlap_words=int(chunking_cfg.get("chunk_overlap_words", 30)),
        strategy=str(chunking_cfg.get("strategy", "fixed")),
    )

    embedder = build_embedder(config)
    model_name = getattr(embedder, "model_name", "unknown")

    # --- embedding cache ---
    safe_name = model_name.replace("/", "_").replace(":", "_")
    cache_path = Path("artifacts/embed_cache") / f"{safe_name}.json"
    cache = _load_embed_cache(cache_path)

    texts = [c["text"] for c in chunks]
    hashes = [_text_hash(t) for t in texts]
    uncached_idx = [i for i, h in enumerate(hashes) if h not in cache]

    if uncached_idx:
        emit(
            f"Embedding {len(uncached_idx)} new chunks "
            f"({len(texts) - len(uncached_idx)} loaded from cache) "
            f"with {model_name}"
        )
        new_vecs = embedder.embed_many([texts[i] for i in uncached_idx])
        for i, vec in zip(uncached_idx, new_vecs):
            cache[hashes[i]] = vec
        _save_embed_cache(cache_path, cache)
    else:
        emit(f"All {len(texts)} chunks loaded from embedding cache ({model_name})")

    matrix = [cache[h] for h in hashes]
    dimensions = len(matrix[0]) if matrix else 0

    emit("Saving vector index")
    index = VectorIndex(matrix=matrix, chunks=chunks, model_name=model_name, dimensions=dimensions)
    index.save(config["paths"]["index_dir"])

    return {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "index_dir": config["paths"]["index_dir"],
        "embedding_provider": config.get("embedding", {}).get("provider", "sentence-transformers"),
        "embedding_model": model_name,
        "chunking_strategy": chunking_cfg.get("strategy", "fixed"),
        "cache_hits": len(texts) - len(uncached_idx),
        "cache_misses": len(uncached_idx),
    }
