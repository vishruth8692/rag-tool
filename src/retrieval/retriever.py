from __future__ import annotations

from src.embeddings.providers import Embedder
from src.index.vector_index import VectorIndex
from src.retrieval.reranker import rerank


class Retriever:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        top_k: int = 4,
        reranker_model: str | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        # Re-ranking is opt-in: set reranker_model to a cross-encoder model name.
        # rerank_candidates controls how many bi-encoder results are fetched before
        # re-ranking (the "candidate pool"). More candidates = better recall ceiling
        # but slower re-ranking. 20 is a sensible default.
        self.reranker_model = reranker_model
        self.rerank_candidates = max(rerank_candidates, top_k)

    def retrieve(self, query: str) -> list[dict]:
        if self.reranker_model:
            # Stage 1: fetch a wider candidate pool from the bi-encoder
            candidates = self.index.search(query_vector=self.embedder.embed(query),
                                            top_k=self.rerank_candidates)
            # Attach chunk text so the cross-encoder can read it
            for c in candidates:
                if "text" not in c:
                    chunk = next(
                        (ch for ch in self.index.chunks if ch["chunk_id"] == c["chunk_id"]),
                        None,
                    )
                    if chunk:
                        c["text"] = chunk["text"]
            # Stage 2: re-rank and return top_k
            return rerank(query, candidates,
                          model_name=self.reranker_model,
                          top_k=self.top_k)
        else:
            return self.index.search(
                query_vector=self.embedder.embed(query),
                top_k=self.top_k,
            )
