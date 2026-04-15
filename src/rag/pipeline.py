from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.embeddings.providers import build_embedder
from src.generation.llm_generator import generate_answer
from src.index.build import build_index
from src.index.vector_index import VectorIndex
from src.retrieval.retriever import Retriever
from src.safety.confidence import compute_confidence, detect_type1, detect_type2_nli


class RAGPipeline:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.embedder = build_embedder(self.config)

        index_dir = self.config["paths"]["index_dir"]
        index_root = Path(index_dir)
        vectors_file = index_root / "vectors.json"
        chunks_file = index_root / "chunks.json"
        if not vectors_file.exists() or not chunks_file.exists():
            try:
                build_index(self.config)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to build/load index in `{index_dir}`. "
                    "Check chunking/embedding configuration and dependencies."
                ) from exc
        if not vectors_file.exists() or not chunks_file.exists():
            raise RuntimeError(
                f"Index files are still missing in `{index_dir}` after build "
                "(expected `vectors.json` and `chunks.json`)."
            )
        self.index = VectorIndex.load(index_dir)

        # Mismatch guard: embedder and index must use the same model
        saved_model = self.index.model_name
        current_model = getattr(self.embedder, "model_name", "")
        if saved_model and current_model and saved_model != current_model:
            raise RuntimeError(
                f"Embedding model mismatch: the index was built with `{saved_model}` "
                f"but the current config uses `{current_model}`. "
                "Re-run `make ingest` (or click Run Eval) to rebuild the index."
            )

        retrieval_cfg = self.config.get("retrieval", {})
        top_k = int(retrieval_cfg.get("top_k", 4))
        reranker_model = retrieval_cfg.get("reranker_model") or None
        rerank_candidates = int(retrieval_cfg.get("rerank_candidates", 20))
        self.retriever = Retriever(
            index=self.index,
            embedder=self.embedder,
            top_k=top_k,
            reranker_model=reranker_model,
            rerank_candidates=rerank_candidates,
        )

    def query(self, text: str) -> dict:
        retrieved = self.retriever.retrieve(text)
        answer = generate_answer(text, retrieved, self.config.get("generation", {}))
        confidence = compute_confidence(retrieved)

        safety_cfg = self.config.get("safety", {})
        type1_threshold = float(safety_cfg.get("low_confidence_threshold", 0.35))
        type2_threshold = float(safety_cfg.get("type2_nli_threshold", 0.5))

        # Pull source texts from the index for NLI grounding check
        source_texts: list[str] = []
        for item in retrieved:
            chunk = next(
                (c for c in self.index.chunks if c["chunk_id"] == item["chunk_id"]),
                None,
            )
            if chunk:
                source_texts.append(chunk["text"])

        type1 = detect_type1(confidence, type1_threshold)

        # ── Type 2 NLI guard ─────────────────────────────────────────────────
        # Skip NLI when:
        #   (a) generation=simple — the answer IS the chunks, so NLI is always ~1.0
        #       and burns compute for nothing.
        #   (b) Type 1 already flagged — retrieval failed; NLI on bad context is noise.
        #   (c) Explicitly disabled in config (safety.skip_type2_nli: true)
        generation_provider = self.config.get("generation", {}).get("provider", "simple")
        skip_type2 = (
            generation_provider == "simple"
            or type1["flagged"]
            or bool(safety_cfg.get("skip_type2_nli", False))
        )
        if skip_type2:
            skip_reason = (
                "simple generation — answer equals retrieved chunks"
                if generation_provider == "simple"
                else "Type 1 flagged — retrieval context unreliable"
                if type1["flagged"]
                else "disabled via config"
            )
            type2 = {
                "flagged": False,
                "score": 1.0,
                "entailed": 0,
                "total_sentences": 0,
                "detail": [],
                "model": "cross-encoder/nli-deberta-v3-small",
                "skipped": True,
                "skip_reason": skip_reason,
            }
        else:
            type2 = detect_type2_nli(answer, source_texts, type2_threshold)

        return {
            "query": text,
            "answer": answer,
            "confidence": confidence,
            "hallucination_risk": type1["flagged"] or type2["flagged"],
            "hallucination": {
                "type1": type1,
                "type2": type2,
            },
            "sources": [
                {
                    "doc_id": item["doc_id"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "chunk_index": item.get("chunk_index"),
                }
                for item in retrieved
            ],
        }
