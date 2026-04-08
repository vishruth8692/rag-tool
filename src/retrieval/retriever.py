from __future__ import annotations

from src.embeddings.providers import Embedder
from src.index.vector_index import VectorIndex


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, top_k: int = 4) -> None:
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.embedder.embed(query)
        return self.index.search(query_vec, self.top_k)
