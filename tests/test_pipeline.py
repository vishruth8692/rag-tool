from pathlib import Path

import pytest

from src.index.vector_index import VectorIndex
from src.rag.pipeline import RAGPipeline


def test_pipeline_query_with_local_index(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)

    matrix = [[1.0, 0.0], [0.0, 1.0]]
    chunks = [
        {"chunk_id": "doc1::chunk::0", "doc_id": "doc1", "chunk_index": 0, "text": "apple banana"},
        {"chunk_id": "doc2::chunk::0", "doc_id": "doc2", "chunk_index": 0, "text": "truck delivery"},
    ]
    VectorIndex(matrix=matrix, chunks=chunks).save(str(index_dir))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  index_dir: {index_dir}",
                "embedding:",
                "  provider: sentence-transformers",
                "  model: sentence-transformers/all-MiniLM-L6-v2",
                "retrieval:",
                "  top_k: 1",
                "generation:",
                "  provider: simple",
                "  max_context_chunks: 1",
                "safety:",
                "  low_confidence_threshold: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    out = RAGPipeline(str(cfg_path)).query("apple")
    assert "answer" in out
    assert isinstance(out["sources"], list)
