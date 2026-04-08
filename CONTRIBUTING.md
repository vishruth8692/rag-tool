# Contributing to RAG Evaluation Framework

Thank you for contributing! This document covers setup, workflow, and code standards.

---

## Setup

```bash
git clone https://github.com/your-org/rag-tool.git
cd rag-tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local LLM testing (hallucination detection):
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b
ollama serve   # keep running in a separate terminal
```

---

## Project layout

```
src/
├── chunking/       Text splitting strategies
├── embeddings/     Embedding model providers (sentence-transformers, openai)
├── ingestion/      Document loading — handles .txt, .csv, .pdf, .json
├── index/          Vector index (build, save, load, mismatch detection)
├── retrieval/      Cosine similarity search
├── generation/     LLM answer generation (simple, ollama, openai, huggingface)
├── safety/         Hallucination detection (Type 1 confidence + Type 2 NLI)
├── eval/           Metrics (Recall@k, MRR, BERTScore, faithfulness) + runner
├── ab_testing/     A/B test runner with bootstrap significance
├── rag/            End-to-end RAGPipeline class
├── dashboard/      Streamlit UI
└── serving/        FastAPI REST server
```

---

## Typical workflow

```bash
# 1. Put documents in data/ (or use the included Wikipedia sample)
# 2. Build index
make ingest

# 3. Run evaluation (uses wikipedia_eval_sample by default)
make eval

# 4. Run A/B test
make ab-test

# 5. Run tests
python -m pytest -q

# 6. Launch dashboard
make dashboard
```

---

## Adding a new chunking strategy

1. Add `chunk_text_<name>()` in `src/chunking/chunker.py`
2. Register it in `chunk_text()` router
3. Add metadata to `_STRATEGY_META` in `src/dashboard/app.py`
4. Add a row to the parameter guide table in the same file
5. Add tests in `tests/test_chunker.py`

## Adding a new embedding provider

1. Add a class implementing `embed_one(text) -> list[float]` and `embed_many(texts) -> list[list[float]]` in `src/embeddings/providers.py`
2. Set `self.model_name` on the class (used for index mismatch detection)
3. Register in `build_embedder()` factory function
4. Update dashboard provider dropdown if it should be user-selectable

## Adding a new generation provider

1. Add a handler in `src/generation/llm_generator.py`
2. Register in `generate_answer()` router
3. Add config section to `configs/base.yaml` under `generation:`

---

## Code standards

- **Config-driven**: new behaviour should be controllable from `configs/base.yaml`, not hard-coded
- **Graceful fallbacks**: all external calls (LLM, NLI model, BERTScore) must fall back cleanly and log a clear message
- **No silent failures**: if a metric can't be computed, store `None` — not `0.0` (which looks like a real zero)
- **Tests for metrics**: `tests/test_metrics.py` — add edge cases for any new metric function
- **Type hints**: use `from __future__ import annotations` and add hints to all public functions

---

## Running tests

```bash
python -m pytest -q                  # all tests
python -m pytest tests/test_chunker.py -v   # one file
```

---

## Pull requests

1. Branch from `main`: `git checkout -b feat/my-feature`
2. Include a short summary of what changed and why
3. Include `python -m pytest -q` output
4. For eval changes, include before/after metric numbers on `wikipedia_eval_sample`
5. Keep PRs focused — one feature or fix per PR

---

## Reporting issues

Please include:
- Python version (`python --version`)
- OS
- Exact error message and traceback
- Config settings (`configs/base.yaml` relevant section)
- Steps to reproduce
