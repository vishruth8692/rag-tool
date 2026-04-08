# RAG Evaluation Framework

An open-source, **product-team-first** framework for building, evaluating, and improving Retrieval-Augmented Generation (RAG) pipelines — with a no-code dashboard built for PMs and a full Python API for engineers.

> **TL;DR for PMs:** Drop your documents in, open the dashboard, pick your goal, click Run. Get retrieval quality scores, hallucination detection, and A/B test results — no code required.

---

## What is RAG?

When you ask an AI assistant a question, it can either answer from memory (which leads to hallucinations) or **look up the answer from your documents first, then respond**. That's RAG — Retrieval-Augmented Generation.

```
Your Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Your Docs  │────▶│  Vector Search   │────▶│  LLM Answer  │
│  (data/)    │     │  (finds top docs)│     │  (grounded)  │
└─────────────┘     └──────────────────┘     └──────────────┘
```

This framework helps you answer: **"How good is my RAG pipeline?"**

---

## What this repo does

| Step | What happens | Where |
|------|-------------|-------|
| 1️⃣  | Drop files into `data/` | `data/` folder |
| 2️⃣  | Build a searchable index | `make ingest` or dashboard |
| 3️⃣  | Run quality evaluation | `make eval` or dashboard |
| 4️⃣  | Compare configurations (A/B) | `make ab-test` or dashboard |
| 5️⃣  | Deploy as an API | `make serve` |

---

## Quickstart (5 minutes)

### Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.com) for local LLM generation

```bash
# 1. Clone and set up
git clone https://github.com/your-org/rag-tool.git
cd rag-tool
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the dashboard
make dashboard
# Opens at http://localhost:8501
```

The repo ships with a **150-article Wikipedia dataset** and **3 evaluation sets** so you can run benchmarks immediately without any data prep.

---

## Dashboard (No-Code)

```bash
make dashboard
# → http://localhost:8501
```

The dashboard walks you through four steps:

### Step 1 — Upload Documents
Drag and drop `.txt`, `.md`, `.csv`, `.pdf`, or `.json` files. The dashboard detects your file types and recommends the best chunking strategy automatically.

### Step 2 — Build Eval Dataset
Create question-answer pairs to test your pipeline. Or use the included Wikipedia eval sets (80 / 519 / 2,461 questions).

### Step 3 — Pick Your Goal and Run

Select what you are trying to measure — the tool sets the right configuration automatically:

| Goal | What gets configured | Time |
|------|---------------------|------|
| 🔍 **Test Retrieval Quality** | Simple generation, 200 queries | ~30s |
| 📊 **Compare Chunking / Embedding** | Simple generation, 150 queries | ~20s |
| 🛡️ **Measure Hallucination** | Ollama LLM, 50 queries | ~5 min |
| 📝 **Evaluate Answer Quality** | Ollama LLM, 80 queries | ~8 min |
| ✅ **Full Validation** | Simple generation, 500 queries | ~2 min |

### Step 4 — Read Results
Results are explained in plain English with colour-coded verdicts (Good / Needs Improvement).

---

## Understanding the Metrics

### Retrieval Metrics
These tell you whether the pipeline **finds the right documents** for a given question.

| Metric | Plain English | Good value |
|--------|--------------|------------|
| **Recall@k** | "Out of every 10 questions, how many times did we pull the right document into the top K results?" | > 0.80 |
| **MRR** (Mean Reciprocal Rank) | "When we find the right document, how high up does it appear? 1st = 1.0, 2nd = 0.5, 3rd = 0.33…" | > 0.70 |

### Faithfulness Metrics
These tell you whether the **answer sticks to what the documents say** (not hallucinated).

| Metric | Plain English | Reliable with? |
|--------|--------------|---------------|
| **Token Overlap** | Word overlap between the answer and retrieved chunks | Ollama / real LLM only |
| **BERTScore** | Semantic similarity between answer and source, using AI embeddings | Ollama / real LLM only |

> ⚠️ Both faithfulness metrics are inflated (close to 1.0) when using `simple` generation because the answer *is* the retrieved text. Use Ollama for meaningful faithfulness scores.

### Hallucination Detection

The framework distinguishes two types of hallucination:

#### Type 1 — Retrieval Failure
The retrieved documents are **not relevant** to the question. The LLM has nothing good to answer from and will likely generate from memory.

- Measured by: retrieval **confidence score**
- Flag threshold: confidence < 0.35 (configurable)
- Fix: improve chunking strategy, embedding model, or add more documents

#### Type 2 — Generation Unfaithfulness
The LLM **makes claims not supported** by the retrieved documents.

- Measured by: NLI (Natural Language Inference) cross-encoder model
- Checks each sentence in the answer against the retrieved context
- Score: fraction of sentences that are *entailed* by the context (1.0 = fully grounded)
- Requires Ollama generation (skipped automatically for `simple`)
- Fix: lower temperature, add system prompts, improve generation config

---

## Chunking Strategies

Chunking is how documents are split before being indexed. The right strategy depends on your data.

| My data is… | Use this strategy | Why |
|-------------|------------------|-----|
| CSV with Q + A columns | `qa_pair` | Each row is already a semantic unit — don't split it |
| Wikipedia / news / blog posts | `paragraph` | Natural paragraph breaks carry complete ideas |
| Policy docs / legal / manuals | `sentence` | Sentence boundaries preserve meaning better than word count |
| Anything / unsure | `fixed` | Reliable baseline for any content type |

### CSV Q&A Format

If your CSV has question and answer columns, set this in `configs/base.yaml`:

```yaml
chunking:
  strategy: qa_pair

data:
  csv_format: qa_pairs
  question_column: question   # or 0-based index: "0"
  answer_column: answer       # or 0-based index: "1"
  pairs_per_chunk: 1          # rows per chunk (1 = maximum precision)
```

Example — what each row becomes as a chunk:
```
Q: What causes lightning?
A: Lightning is caused by electrical discharge between regions of differing charge...
```

---

## Configuration

All settings live in `configs/base.yaml`. The dashboard writes temporary override configs — you don't need to edit YAML to use the dashboard.

```yaml
chunking:
  strategy: fixed          # fixed | sentence | paragraph | qa_pair
  chunk_size_words: 180
  chunk_overlap_words: 30

embedding:
  provider: sentence-transformers   # free, local
  sentence_transformers:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    # Other free models:
    # BAAI/bge-small-en-v1.5           (better retrieval quality)
    # multi-qa-MiniLM-L6-cos-v1        (trained on Q&A, best for RAG)

generation:
  provider: simple         # simple | ollama | openai | huggingface
  ollama:
    model: llama3.1:8b
```

### Generation Providers

| Provider | Cost | Speed | Use for |
|----------|------|-------|---------|
| `simple` | Free | Instant | Retrieval benchmarking, chunking/embedding comparison |
| `ollama` | Free (local) | Slow | Hallucination detection, answer quality evaluation |
| `openai` | Paid | Fast | Production answer quality benchmarks |
| `huggingface` | Free (API) | Varies | Experimenting with open-source LLMs |

---

## Running from the Command Line

```bash
# Build the vector index from data/
make ingest

# Run evaluation against the sample dataset
make eval

# Run A/B test comparing two configs
make ab-test

# Start the REST API server
make serve

# Run tests
python -m pytest -q
```

Override defaults:
```bash
make eval DATASET=eval/datasets/wikipedia_eval_hard.jsonl
make ab-test CONFIG_A=configs/rag_a.yaml CONFIG_B=configs/rag_b.yaml
```

---

## REST API

Start the server:
```bash
make serve  # default: http://0.0.0.0:8000
```

**Query endpoint:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes lightning?"}'
```

Response:
```json
{
  "query": "What causes lightning?",
  "answer": "Lightning is caused by...",
  "confidence": 0.87,
  "hallucination_risk": false,
  "hallucination": {
    "type1": { "flagged": false, "risk_level": "low", "confidence": 0.87 },
    "type2": { "flagged": false, "score": 0.92, "skipped": false }
  },
  "sources": [
    { "doc_id": "lightning.txt", "chunk_id": "lightning.txt::chunk::2", "score": 0.91 }
  ]
}
```

---

## Project Structure

```
rag-tool/
├── data/                        # Drop your documents here
│   └── wikipedia/               # Sample dataset (150 Wikipedia articles)
├── eval/
│   └── datasets/                # Evaluation question sets
│       ├── wikipedia_eval_sample.jsonl   (80 questions — fast iteration)
│       ├── wikipedia_eval_hard.jsonl     (519 questions — stress test)
│       └── wikipedia_eval_full.jsonl     (2,461 questions — final validation)
├── configs/
│   ├── base.yaml                # Main configuration
│   ├── rag_a.yaml               # Config A for A/B testing
│   └── rag_b.yaml               # Config B for A/B testing
├── src/
│   ├── chunking/                # Text splitting strategies
│   ├── embeddings/              # Embedding model providers
│   ├── ingestion/               # Document loading (txt, csv, pdf, json)
│   ├── index/                   # Vector index build + save/load
│   ├── retrieval/               # Similarity search
│   ├── generation/              # LLM answer generation
│   ├── safety/                  # Hallucination detection (Type 1 + Type 2 NLI)
│   ├── eval/                    # Metrics + evaluation runner
│   ├── ab_testing/              # A/B test runner + bootstrap significance
│   ├── rag/                     # End-to-end pipeline
│   ├── dashboard/               # Streamlit UI
│   └── serving/                 # FastAPI REST server
├── scripts/                     # CLI entry points
├── tests/                       # Unit tests
├── Makefile                     # One-command workflows
└── requirements.txt
```

---

## Eval Dataset Format

Evaluation datasets are `.jsonl` files — one question per line:

```jsonl
{"query": "What is photosynthesis?", "reference_answer": "...", "expected_doc_ids": ["biology.txt"], "difficulty": "easy"}
{"query": "Compare C3 and C4 photosynthesis", "reference_answer": "...", "expected_doc_ids": ["biology.txt"], "difficulty": "hard"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `query` | ✅ | The question |
| `expected_doc_ids` | ✅ | Which document(s) should be retrieved (for Recall@k / MRR) |
| `reference_answer` | Optional | Gold answer for faithfulness comparison |
| `difficulty` | Optional | `easy` / `medium` / `hard` — used for dataset filtering |

---

## Local LLM Setup (Ollama)

Required for hallucination detection and answer quality evaluation.

```bash
# Install (macOS)
brew install ollama
# OR download from https://ollama.com

# Pull the model used in default config
ollama pull llama3.1:8b

# Start the server (keep this running in a terminal)
ollama serve
```

Then in `configs/base.yaml`:
```yaml
generation:
  provider: ollama
  ollama:
    model: llama3.1:8b
```

Or just select **🛡️ Measure Hallucination** in the dashboard — it sets this automatically.

---

## A/B Testing

The A/B test runs the same eval dataset through two different configurations and uses **bootstrap statistical significance** to tell you whether the difference is real.

```bash
# Edit configs/rag_a.yaml and configs/rag_b.yaml with different settings, then:
make ab-test
```

Or use the dashboard → **A/B Compare** mode — configure Control and Test side by side.

Results include:
- Per-metric delta (Test − Control)
- Bootstrap p-value (< 0.05 = statistically significant)
- Per-query retrieval trace for debugging

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
