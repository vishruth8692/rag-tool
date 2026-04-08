from __future__ import annotations

import os
from typing import Any

from src.common.http_client import post_json


def _simple_generate(contexts: list[dict], max_context_chunks: int) -> str:
    if not contexts:
        return "I do not have enough context to answer this confidently."

    selected = contexts[:max_context_chunks]
    snippets = [ctx["text"].strip()[:320] for ctx in selected]
    return "Answer based on retrieved context: " + " ".join(snippets)


def _build_prompt(query: str, contexts: list[dict], max_context_chunks: int) -> str:
    selected = contexts[:max_context_chunks]
    blocks = []
    for i, ctx in enumerate(selected, start=1):
        blocks.append(f"[{i}] doc_id={ctx['doc_id']} chunk_id={ctx['chunk_id']}\n{ctx['text']}")

    context_text = "\n\n".join(blocks)
    return (
        "You are a grounded RAG assistant. Use only the provided context. "
        "If the answer is not in context, say you do not know. "
        "Answer in 3-6 sentences and cite sources like [1], [2].\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )


def _generate_openai(prompt: str, cfg: dict[str, Any]) -> str:
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var `{api_key_env}` for OpenAI generation provider.")

    model = cfg.get("model", "gpt-4o-mini")
    base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    timeout = int(cfg.get("timeout", 90))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer using provided context only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(cfg.get("temperature", 0.1)),
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    res = post_json(url=f"{base_url}/chat/completions", payload=payload, headers=headers, timeout=timeout)
    return res["choices"][0]["message"]["content"].strip()


def _generate_ollama(prompt: str, cfg: dict[str, Any]) -> str:
    model = cfg.get("model", "llama3.1:8b")
    base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
    timeout = int(cfg.get("timeout", 120))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(cfg.get("temperature", 0.1))},
    }
    res = post_json(url=f"{base_url}/api/generate", payload=payload, timeout=timeout)
    return str(res.get("response", "")).strip()


def _generate_huggingface(prompt: str, cfg: dict[str, Any]) -> str:
    token_env = cfg.get("api_key_env", "HF_API_TOKEN")
    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(f"Missing env var `{token_env}` for HuggingFace generation provider.")

    model = cfg.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
    timeout = int(cfg.get("timeout", 120))
    endpoint = cfg.get("endpoint", f"https://api-inference.huggingface.co/models/{model}")

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(cfg.get("max_new_tokens", 280)),
            "temperature": float(cfg.get("temperature", 0.1)),
            "return_full_text": False,
        },
    }
    headers = {"Authorization": f"Bearer {token}"}

    res = post_json(url=endpoint, payload=payload, headers=headers, timeout=timeout)

    if isinstance(res, list) and res and isinstance(res[0], dict):
        return str(res[0].get("generated_text", "")).strip()
    if isinstance(res, dict) and "generated_text" in res:
        return str(res["generated_text"]).strip()

    raise RuntimeError(f"Unexpected HuggingFace response: {res}")


def generate_answer(query: str, contexts: list[dict], generation_cfg: dict[str, Any]) -> str:
    max_context_chunks = int(generation_cfg.get("max_context_chunks", 3))
    provider = generation_cfg.get("provider", "simple")
    fallback_to_simple = bool(generation_cfg.get("fallback_to_simple", True))

    if provider == "simple":
        return _simple_generate(contexts, max_context_chunks)

    prompt = _build_prompt(query=query, contexts=contexts, max_context_chunks=max_context_chunks)

    try:
        if provider == "openai":
            return _generate_openai(prompt, generation_cfg.get("openai", {}))
        if provider == "ollama":
            return _generate_ollama(prompt, generation_cfg.get("ollama", {}))
        if provider == "huggingface":
            return _generate_huggingface(prompt, generation_cfg.get("huggingface", {}))
        raise ValueError(f"Unsupported generation provider: {provider}")
    except Exception as exc:
        if fallback_to_simple:
            base = _simple_generate(contexts, max_context_chunks)
            return f"{base}\n\n[Fallback reason: {exc}]"
        raise
