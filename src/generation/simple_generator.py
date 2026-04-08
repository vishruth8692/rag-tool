from __future__ import annotations


def generate_answer(query: str, contexts: list[dict], max_context_chunks: int = 3) -> str:
    if not contexts:
        return "I do not have enough context to answer this confidently."

    selected = contexts[:max_context_chunks]
    snippets = []
    for ctx in selected:
        text = ctx["text"].strip()
        snippets.append(text[:240])

    combined = " ".join(snippets)
    return f"Answer based on retrieved context: {combined}"
