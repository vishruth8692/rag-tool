from src.chunking.chunker import chunk_text


def test_chunk_text_non_empty() -> None:
    text = " ".join(["token"] * 300)
    chunks = chunk_text(text, chunk_size_words=100, chunk_overlap_words=20)
    assert len(chunks) >= 3
