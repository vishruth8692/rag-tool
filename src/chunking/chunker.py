from __future__ import annotations

import re

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text_fixed(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size_words - chunk_overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def chunk_text_sentence(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return chunk_text_fixed(text, chunk_size_words, chunk_overlap_words)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current and current_words + s_words > chunk_size_words:
            chunks.append(" ".join(current))
            if chunk_overlap_words > 0:
                overlap_words = " ".join(current).split()[-chunk_overlap_words:]
                current = [" ".join(overlap_words)] if overlap_words else []
                current_words = len(overlap_words)
            else:
                current = []
                current_words = 0
        current.append(sentence)
        current_words += s_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def chunk_text_paragraph(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return chunk_text_fixed(text, chunk_size_words, chunk_overlap_words)

    chunks: list[str] = []
    for para in paragraphs:
        if len(para.split()) <= chunk_size_words:
            chunks.append(para)
        else:
            chunks.extend(chunk_text_fixed(para, chunk_size_words, chunk_overlap_words))
    return chunks


def chunk_text(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    strategy: str = "fixed",
) -> list[str]:
    strategy = strategy.lower().strip()
    if strategy == "fixed":
        return chunk_text_fixed(text, chunk_size_words, chunk_overlap_words)
    if strategy in {"sentence", "semantic"}:
        return chunk_text_sentence(text, chunk_size_words, chunk_overlap_words)
    if strategy == "paragraph":
        return chunk_text_paragraph(text, chunk_size_words, chunk_overlap_words)
    if strategy == "qa_pair":
        # Each doc is already a pre-formed Q+A chunk — return as-is.
        # Grouping is handled at load time (pairs_per_chunk in data config).
        return [text] if text.strip() else []
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def chunk_documents(
    docs: list[dict],
    chunk_size_words: int,
    chunk_overlap_words: int,
    strategy: str = "fixed",
) -> list[dict]:
    chunked: list[dict] = []
    for doc in docs:
        pieces = chunk_text(
            doc["text"],
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            strategy=strategy,
        )
        for idx, chunk in enumerate(pieces):
            chunked.append(
                {
                    "chunk_id": f"{doc['doc_id']}::chunk::{idx}",
                    "doc_id": doc["doc_id"],
                    "chunk_index": idx,
                    "text": chunk,
                    "path": doc["path"],
                }
            )
    return chunked
