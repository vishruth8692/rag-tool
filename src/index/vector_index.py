from __future__ import annotations

import json
import time
from pathlib import Path


class VectorIndex:
    def __init__(
        self,
        matrix: list[list[float]],
        chunks: list[dict],
        model_name: str = "",
        dimensions: int = 0,
    ) -> None:
        self.matrix = matrix
        self.chunks = chunks
        self.model_name = model_name
        self.dimensions = dimensions or (len(matrix[0]) if matrix else 0)
        self.meta: dict = {}

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        if not self.matrix:
            return []
        sims = [self._dot(row, query_vector) for row in self.matrix]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]

        results: list[dict] = []
        for idx in top_idx:
            item = dict(self.chunks[idx])
            item["score"] = float(sims[idx])
            results.append(item)
        return results

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vectors.json").write_text(json.dumps(self.matrix), encoding="utf-8")
        (out / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=True, indent=2), encoding="utf-8")
        meta = {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "num_chunks": len(self.chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str) -> "VectorIndex":
        root = Path(index_dir)
        matrix = json.loads((root / "vectors.json").read_text(encoding="utf-8"))
        chunks = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
        meta_path = root / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        inst = cls(
            matrix=matrix,
            chunks=chunks,
            model_name=meta.get("model_name", ""),
            dimensions=meta.get("dimensions", len(matrix[0]) if matrix else 0),
        )
        inst.meta = meta
        return inst
