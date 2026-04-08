from __future__ import annotations

import os
from typing import Protocol

from src.common.http_client import post_json


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers provider requires `sentence-transformers` package."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.model.encode(texts, normalize_embeddings=True)
        return matrix.tolist()


class OpenAIEmbedder:
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var `{api_key_env}` for OpenAI embeddings provider.")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        res = post_json(
            url=f"{self.base_url}/embeddings",
            payload=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return res["data"][0]["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        res = post_json(
            url=f"{self.base_url}/embeddings",
            payload=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return [item["embedding"] for item in res["data"]]


def build_embedder(config: dict) -> Embedder:
    embedding_cfg = config.get("embedding", {})
    provider = embedding_cfg.get("provider", embedding_cfg.get("type", "sentence-transformers"))
    generic_model = embedding_cfg.get("model")
    if provider == "hash":
        raise ValueError("`hash` embedding provider has been removed. Use `sentence-transformers` or `openai`.")

    if provider == "sentence-transformers":
        st_cfg = embedding_cfg.get("sentence_transformers", {})
        model_name = st_cfg.get("model_name", generic_model or "sentence-transformers/all-MiniLM-L6-v2")
        return SentenceTransformerEmbedder(model_name=model_name)

    if provider == "openai":
        o_cfg = embedding_cfg.get("openai", {})
        return OpenAIEmbedder(
            model=o_cfg.get("model", generic_model or "text-embedding-3-small"),
            api_key_env=o_cfg.get("api_key_env", "OPENAI_API_KEY"),
            base_url=o_cfg.get("base_url", "https://api.openai.com/v1"),
            timeout=int(o_cfg.get("timeout", 60)),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
