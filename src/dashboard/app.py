from __future__ import annotations

import csv
import glob
import io
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from src.eval.runner import run_eval
from src.ab_testing.runner import run_ab
from src.common.config import load_config
from src.rag.pipeline import RAGPipeline

import yaml


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Eval objectives ───────────────────────────────────────────────────────────
# Each objective is a preset that drives: generation provider, max queries,
# preferred dataset, and which metrics to highlight.

_OBJECTIVES: dict[str, dict] = {
    "🔍 Test Retrieval Quality": {
        "provider": "simple",
        "max_queries": 200,
        "dataset_keyword": "hard",
        "focus_metrics": ["avg_recall_at_k", "avg_mrr"],
        "description": (
            "Are we finding the right documents? "
            "Uses **simple** generation (no LLM needed) so it runs in seconds. "
            "Primary metrics: **Recall@k** and **MRR**."
        ),
        "provider_reason": "Retrieval metrics don't need an LLM — `simple` is instant.",
    },
    "📊 Compare Chunking / Embedding": {
        "provider": "simple",
        "max_queries": 150,
        "dataset_keyword": "hard",
        "focus_metrics": ["avg_recall_at_k", "avg_mrr", "avg_latency_ms"],
        "description": (
            "Which chunking strategy or embedding model retrieves better? "
            "Uses **simple** generation for speed — run A/B mode for side-by-side comparison. "
            "Primary metrics: **Recall@k**, **MRR**, **Latency**."
        ),
        "provider_reason": "Chunking/embedding differences show up in retrieval, not generation.",
    },
    "🛡️ Measure Hallucination": {
        "provider": "ollama",
        "max_queries": 50,
        "dataset_keyword": "sample",
        "focus_metrics": ["type1_hallucination_rate", "type2_hallucination_rate", "avg_type2_nli_score"],
        "description": (
            "Is the model making things up? "
            "Uses **ollama** (local LLM) so the answer is a real generated response — "
            "required for NLI-based Type 2 hallucination detection. "
            "Primary metrics: **Type 1 rate**, **Type 2 rate**, **NLI Score**."
        ),
        "provider_reason": "NLI hallucination check requires a real LLM answer, not a chunk copy.",
    },
    "📝 Evaluate Answer Quality": {
        "provider": "ollama",
        "max_queries": 80,
        "dataset_keyword": "sample",
        "focus_metrics": ["avg_faithfulness_bertscore", "avg_mrr", "avg_latency_ms"],
        "description": (
            "How faithful and relevant are the generated answers? "
            "Uses **ollama** so BERTScore measures real answer quality. "
            "Primary metrics: **BERTScore Faithfulness**, **MRR**, **Latency**."
        ),
        "provider_reason": "BERTScore faithfulness is only meaningful with a real LLM answer.",
    },
    "✅ Full Validation": {
        "provider": "simple",
        "max_queries": 500,
        "dataset_keyword": "full",
        "focus_metrics": ["avg_recall_at_k", "avg_mrr", "avg_faithfulness_proxy", "hallucination_rate"],
        "description": (
            "Final sign-off before production. Runs a large query sample across all metrics. "
            "Uses **simple** generation for speed — switch to ollama for answer quality checks. "
            "Primary metrics: **all**."
        ),
        "provider_reason": "Large-scale validation — simple keeps it fast for the retrieval layer.",
    },
}


def _render_objective_selector(key_prefix: str, dataset_files: list[str]) -> dict:
    """Render the objective card and return resolved settings.

    Returns dict with keys:
        provider_override, max_queries, selected_dataset, focus_metrics, objective_name
    """
    st.markdown("#### What is your goal for this run?")
    objective_names = list(_OBJECTIVES.keys())
    chosen = st.selectbox(
        "Objective",
        options=objective_names,
        index=0,
        key=f"{key_prefix}_objective",
        help="Pick your goal and we'll automatically set the generation provider, query count, and dataset.",
    )
    obj = _OBJECTIVES[chosen]

    # Show description card
    st.info(f"{obj['description']}")

    # Provider badge
    provider = obj["provider"]
    badge = "⚡ `simple` — instant, no LLM" if provider == "simple" else "🦙 `ollama` — local LLM required"
    st.caption(f"**Generation provider auto-set to:** {badge}  ·  *{obj['provider_reason']}*")

    # Warn if ollama selected but may not be running
    if provider == "ollama":
        st.warning(
            "🦙 **ollama required** — make sure it is running: `ollama serve` in a terminal. "
            "If ollama is not installed, switch to a retrieval-only objective instead."
        )

    # Dataset — prefer the keyword match, fallback to first
    keyword = obj["dataset_keyword"]
    preferred_ds = next((p for p in dataset_files if keyword in Path(p).name), dataset_files[0] if dataset_files else None)

    col_ds, col_mq = st.columns([3, 1])
    with col_ds:
        selected_dataset = st.selectbox(
            "Evaluation Dataset",
            options=dataset_files,
            format_func=lambda p: f"{Path(p).name}  ({_count_jsonl_rows(p)} queries)",
            index=dataset_files.index(preferred_ds) if preferred_ds in dataset_files else 0,
            key=f"{key_prefix}_obj_dataset",
            help="Auto-selected based on your objective. You can override this.",
        )
    with col_mq:
        dataset_size = _count_jsonl_rows(selected_dataset) if selected_dataset else 100
        max_q = st.number_input(
            "Max queries",
            min_value=10,
            max_value=dataset_size or 2500,
            value=min(obj["max_queries"], dataset_size or obj["max_queries"]),
            step=10,
            key=f"{key_prefix}_obj_max_q",
            help=f"Suggested {obj['max_queries']} for this objective. Dataset has {dataset_size} total.",
        )

    # Overheat warning for large ollama runs
    if provider == "ollama" and max_q > 100:
        st.warning(
            f"⚠️ **{max_q} queries with ollama** may overheat your laptop. "
            "Consider keeping it at ≤ 100 for hallucination/quality objectives."
        )

    # Focus metrics hint
    st.caption(f"📌 **Focus on these metrics in the results:** `{'`  ·  `'.join(obj['focus_metrics'])}`")

    return {
        "provider_override": provider,
        "max_queries": int(max_q),
        "selected_dataset": selected_dataset,
        "focus_metrics": obj["focus_metrics"],
        "objective_name": chosen,
    }


# ── Data advisor helpers ──────────────────────────────────────────────────────

def _scan_data_folder(data_dir: str = "data") -> dict:
    """Return counts of file types found in data_dir (recursively)."""
    root = Path(data_dir)
    counts: dict[str, int] = {}
    total_size_kb = 0.0
    if not root.exists():
        return {"counts": counts, "total_size_kb": 0, "files": []}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            ext = p.suffix.lower() or "(no ext)"
            counts[ext] = counts.get(ext, 0) + 1
            total_size_kb += p.stat().st_size / 1024
            files.append(p)
    return {"counts": counts, "total_size_kb": total_size_kb, "files": files}


def _recommend_strategy(counts: dict[str, int]) -> tuple[str, str]:
    """Return (recommended_strategy, explanation) based on file types found."""
    has_csv = counts.get(".csv", 0) > 0
    has_txt = counts.get(".txt", 0) > 0
    has_md = counts.get(".md", 0) > 0
    has_pdf = counts.get(".pdf", 0) > 0
    only_csv = has_csv and not has_txt and not has_md and not has_pdf

    if only_csv:
        return (
            "qa_pair",
            "Only CSV files detected. If your CSV has question + answer columns, "
            "**qa_pair** is the best strategy — each row becomes one precise chunk.",
        )
    if has_txt or has_md:
        return (
            "paragraph",
            "Text/Markdown files detected. **paragraph** works well for articles, "
            "Wikipedia-style text, and structured long-form documents.",
        )
    if has_pdf:
        return (
            "sentence",
            "PDF files detected. **sentence** is recommended — PDFs often have "
            "inconsistent paragraph breaks, so sentence-level grouping is more reliable.",
        )
    if has_csv:
        return (
            "qa_pair",
            "CSV file detected alongside other formats. Use **qa_pair** if the CSV "
            "has Q&A columns, or **fixed** if it is tabular data without clear Q&A structure.",
        )
    return ("fixed", "No files detected yet. **fixed** is a safe default for any content type.")


def _peek_csv(path: Path, n_rows: int = 3) -> tuple[list[str], list[list[str]]]:
    """Return (header, first_n_data_rows) from a CSV file."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(c.strip() for c in r)]
        if not rows:
            return [], []
        header = rows[0]
        data = rows[1 : 1 + n_rows]
        return header, data
    except Exception:
        return [], []


def _build_qa_chunk_preview(
    header: list[str],
    data_rows: list[list[str]],
    q_col_spec: str,
    a_col_spec: str,
) -> list[str]:
    """Build example chunks from CSV preview rows using the given column specs."""
    if not header or not data_rows:
        return []

    def resolve(spec: str) -> int | None:
        try:
            idx = int(spec)
            return idx if 0 <= idx < len(header) else None
        except ValueError:
            for i, h in enumerate(header):
                if h.lower().strip() == spec.lower().strip():
                    return i
            return None

    q_idx = resolve(q_col_spec)
    a_idx = resolve(a_col_spec)
    if q_idx is None or a_idx is None:
        return []

    chunks = []
    for row in data_rows:
        q = row[q_idx].strip() if q_idx < len(row) else ""
        a = row[a_idx].strip() if a_idx < len(row) else ""
        if q or a:
            chunks.append(f"Q: {q}\nA: {a}")
    return chunks


def _render_data_advisor(data_dir: str = "data") -> None:
    """Render the data folder summary + strategy recommendation card."""
    info = _scan_data_folder(data_dir)
    counts = info["counts"]
    size_kb = info["total_size_kb"]
    total_files = sum(counts.values())

    if total_files == 0:
        st.warning("📂 No files found in `data/`. Upload files in Step 1 before running eval.")
        return

    # File summary
    ext_summary = "  ·  ".join(
        f"`{ext}` × {n}" for ext, n in sorted(counts.items())
    )
    st.caption(f"📂 **{total_files} files** in `data/`  —  {ext_summary}  —  {size_kb:.0f} KB total")

    strategy, explanation = _recommend_strategy(counts)
    st.info(f"💡 **Recommended strategy: `{strategy}`** — {explanation}")


def _render_chunk_preview(data_dir: str, chunking_strategy: str,
                           q_col: str, a_col: str) -> None:
    """Render a live preview of what chunks will look like from the actual data."""
    root = Path(data_dir)
    if not root.exists():
        return

    with st.expander("🔍 Live Chunk Preview — see what your data will look like as chunks", expanded=False):
        if chunking_strategy == "qa_pair":
            csv_files = [p for p in root.rglob("*.csv") if not p.name.startswith(".")]
            if not csv_files:
                st.info("No CSV files found in `data/`. Upload a CSV to see a preview.")
                return

            file_to_preview = csv_files[0]
            st.caption(f"Previewing: `{file_to_preview.name}` (first file found)")
            header, data_rows = _peek_csv(file_to_preview, n_rows=3)

            if not header:
                st.warning("Could not read CSV — check encoding or file content.")
                return

            # Show raw CSV table
            st.markdown("**Raw CSV (first 3 rows):**")
            try:
                preview_df = pd.DataFrame(data_rows, columns=header[:len(data_rows[0])] if data_rows else header)
                st.dataframe(preview_df, use_container_width=True)
            except Exception:
                st.code("\n".join([" | ".join(header)] + [" | ".join(r) for r in data_rows]))

            # Show what chunks will look like
            chunks = _build_qa_chunk_preview(header, data_rows, q_col, a_col)
            if chunks:
                st.markdown("**→ Chunks that will be embedded:**")
                for i, chunk in enumerate(chunks, 1):
                    st.code(chunk, language=None)
            else:
                st.warning(
                    f"Could not build chunk preview — column spec `{q_col}` or `{a_col}` "
                    f"not found in headers: `{header}`. Check your column names above."
                )

        elif chunking_strategy in ("fixed", "sentence", "paragraph"):
            # Show a text file preview
            txt_files = [p for p in root.rglob("*")
                         if p.suffix.lower() in (".txt", ".md") and not p.name.startswith(".")]
            if not txt_files:
                st.info("No text files found — upload a `.txt` or `.md` file to see a preview.")
                return
            file_to_preview = txt_files[0]
            st.caption(f"Previewing: `{file_to_preview.name}`")
            sample = file_to_preview.read_text(encoding="utf-8", errors="ignore")[:800]

            # Show a few chunks using the real chunker
            try:
                from src.chunking.chunker import chunk_text
                chunks = chunk_text(sample, chunk_size_words=180,
                                    chunk_overlap_words=30, strategy=chunking_strategy)
                st.markdown("**→ First 2 chunks that will be embedded:**")
                for chunk in chunks[:2]:
                    st.code(chunk, language=None)
            except Exception as exc:
                st.warning(f"Preview failed: {exc}")

        else:
            st.info("Select a strategy to see a chunk preview.")


def _read_cfg(path: str) -> dict:
    return load_config(path)


def _list_data_files(data_dir: str = "data") -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_file()])


def _save_uploaded_files(uploaded_files, data_dir: str = "data") -> list[str]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for uploaded in uploaded_files:
        dest = root / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        saved.append(uploaded.name)
    return saved


def _count_jsonl_rows(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _dataset_selectbox(dataset_files: list[str], key: str) -> str:
    """Render a dataset dropdown with query counts in the label."""
    labels = {p: f"{Path(p).name}  ({_count_jsonl_rows(p)} queries)" for p in dataset_files}
    preferred = next((p for p in dataset_files if "sample" in p), dataset_files[0])
    selected = st.selectbox(
        "Evaluation Dataset",
        options=dataset_files,
        format_func=lambda p: labels[p],
        index=dataset_files.index(preferred),
        key=key,
        help="Choose which question set to evaluate against. Larger datasets give more reliable metrics but take longer to run.",
    )
    return selected


def _save_eval_dataset(rows: list[dict], out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(path)


# Per-strategy behaviour — drives labels, help text, and whether overlap is active
_STRATEGY_META = {
    "qa_pair": {
        "description": "For CSV datasets with question + answer columns. Each row becomes one chunk — no splitting. Set CSV Format below to 'qa_pairs' and specify your column names.",
        "chunk_size_label": "Chunk Size (words) — not used",
        "chunk_size_help": "Not used in qa_pair mode — each Q+A row is already the chunk.",
        "overlap_active": False,
        "overlap_help": "Not used in qa_pair mode.",
    },
    "fixed": {
        "description": "Splits every N words regardless of sentence boundaries. Fast and predictable. Best baseline.",
        "chunk_size_label": "Chunk Size (words)",
        "chunk_size_help": "Every chunk will contain exactly this many words (except the last). Try 100–200.",
        "overlap_active": True,
        "overlap_help": "Words shared between adjacent chunks. Helps preserve context at split boundaries. Typically 10–20% of chunk size.",
    },
    "sentence": {
        "description": "Groups whole sentences together until the word limit is reached. Preserves sentence meaning — no mid-sentence cuts.",
        "chunk_size_label": "Max Words Per Chunk",
        "chunk_size_help": "Sentences accumulate into a chunk until this word count is hit, then a new chunk starts. Actual chunk size may be slightly under this limit.",
        "overlap_active": True,
        "overlap_help": "Takes the last N words from the previous chunk as a prefix. Warning: this can create partial sentence fragments at chunk boundaries.",
    },
    "paragraph": {
        "description": "Keeps each paragraph as one whole chunk. Only falls back to word-splitting if a paragraph is unusually long.",
        "chunk_size_label": "Fallback Size (words)",
        "chunk_size_help": "Only triggers when a paragraph exceeds this word count. Most Wikipedia paragraphs are 80–120 words, so this rarely applies with the default of 180.",
        "overlap_active": False,
        "overlap_help": "Not used in paragraph mode — each paragraph is already a natural unit. Overlap only applies in the fixed-size fallback path.",
    },
    "semantic": {
        "description": "Groups sentences by meaning similarity. Produces the most coherent chunks but requires more compute.",
        "chunk_size_label": "Max Words Per Chunk",
        "chunk_size_help": "Upper bound on chunk size. Semantic grouping may produce chunks smaller than this limit.",
        "overlap_active": False,
        "overlap_help": "Not used in semantic mode — chunk boundaries are determined by meaning, not word position.",
    },
}

_PARAM_GUIDE_TABLE = """\
| Parameter | fixed | sentence | paragraph | qa_pair | semantic |
|---|---|---|---|---|---|
| **Chunk Size** | Hard split every N words | Word threshold before new chunk | Fallback only — rarely triggers | **Ignored** — rows are chunks | Upper bound only |
| **Overlap** | Active — shared words at boundaries | Active — but may fragment sentences | **Ignored** | **Ignored** | **Ignored** |
| **Top K** | Number of chunks retrieved and passed to the LLM — active in all modes | | | | |

**When to use each strategy:**
- **fixed** — baseline, works everywhere, start here
- **sentence** — policy/support docs with clear sentences
- **paragraph** — Wikipedia, articles, structured long-form text
- **qa_pair** — CSV files with question + answer columns (FAQ, support, exam datasets)
- **semantic** — highest quality, needs more compute, good for technical docs
"""


def _format_strategy(s: str) -> str:
    if s == "semantic":
        return "semantic — coming soon"
    return s


def _config_editor(title: str, cfg: dict, key_prefix: str) -> dict:
    st.markdown(f"**{title}**")

    # ── Section 1: Chunking ───────────────────────────────────────────────────
    st.markdown("**Chunking**")

    # Data advisor — scan data/ and suggest a strategy
    _render_data_advisor()

    # Strategy chooser card — helps PMs pick without reading docs
    with st.expander("📖 What strategy should I use?", expanded=False):
        st.markdown(
            "| My data looks like… | Recommended strategy | Why |\n"
            "|---|---|---|\n"
            "| **CSV with Q column + A column** (FAQ, support tickets, exam prep) | `qa_pair` | Each row is already the perfect semantic unit — don't split it |\n"
            "| **Wikipedia / news articles / blog posts** | `paragraph` | Natural paragraph breaks carry full ideas |\n"
            "| **Policy docs / legal text / technical manuals** | `sentence` | Dense prose — sentence boundaries preserve meaning better than word count |\n"
            "| **Anything else / unsure** | `fixed` | Reliable baseline that works across all content types |\n"
            "| **Mixed formats (CSV + text)** | `paragraph` or `fixed` | Use `qa_pair` only when ALL files in data/ are Q&A CSVs |\n"
        )
        st.caption(
            "💡 **Rule of thumb:** Start with `fixed`. "
            "Switch to `paragraph` for articles. "
            "Switch to `qa_pair` only when your entire `data/` folder is Q&A CSV files."
        )

    strategy_options = ["fixed", "sentence", "paragraph", "qa_pair", "semantic"]
    strategy_default = str(cfg.get("chunking", {}).get("strategy", "fixed"))
    chunking_strategy = st.selectbox(
        "Strategy",
        options=strategy_options,
        format_func=_format_strategy,
        index=strategy_options.index(strategy_default) if strategy_default in strategy_options else 0,
        key=f"{key_prefix}_chunk_strategy",
        help="Controls how documents are split before embedding. Each strategy treats Chunk Size and Overlap differently — see the Parameter Guide above.",
    )

    meta = _STRATEGY_META.get(chunking_strategy, _STRATEGY_META["fixed"])
    st.caption(meta["description"])

    if chunking_strategy == "semantic":
        st.warning("Semantic chunking is coming soon — it requires embedding-level sentence clustering and is not yet implemented. Please select another strategy.")

    cs_col, co_col = st.columns(2)
    chunk_size = int(
        cs_col.number_input(
            meta["chunk_size_label"],
            min_value=1,
            value=int(cfg.get("chunking", {}).get("chunk_size_words", 180)),
            step=10,
            key=f"{key_prefix}_chunk_size",
            help=meta["chunk_size_help"],
        )
    )
    chunk_overlap = int(
        co_col.number_input(
            "Chunk Overlap (words)" if meta["overlap_active"] else "Chunk Overlap — not used",
            min_value=0,
            value=int(cfg.get("chunking", {}).get("chunk_overlap_words", 30)),
            step=5,
            key=f"{key_prefix}_chunk_overlap",
            disabled=not meta["overlap_active"],
            help=meta["overlap_help"],
        )
    )

    if meta["overlap_active"] and chunk_overlap >= chunk_size:
        st.warning(
            f"Overlap ({chunk_overlap}) must be smaller than Chunk Size ({chunk_size}). "
            "It will be auto-clamped when the eval runs."
        )

    # ── CSV Format (shown only when qa_pair or single_col strategy) ───────────
    data_cfg = cfg.get("data", {})
    csv_format_default = str(data_cfg.get("csv_format", "raw"))

    if chunking_strategy == "qa_pair":
        st.markdown("**CSV Format** *(required for qa_pair strategy)*")
        st.caption(
            "Tell the loader which columns hold the question and answer. "
            "Use the column header name (e.g. `question`) or a 0-based index (e.g. `0`)."
        )
        q_col_default = str(data_cfg.get("question_column", "0"))
        a_col_default = str(data_cfg.get("answer_column", "1"))
        pairs_default = int(data_cfg.get("pairs_per_chunk", 1))

        csv_q_col, csv_a_col = st.columns(2)
        csv_question_col = csv_q_col.text_input(
            "Question column",
            value=q_col_default,
            placeholder="e.g. question  or  0",
            key=f"{key_prefix}_csv_q_col",
            help="Column header name or 0-based index for the question field.",
        )
        csv_answer_col = csv_a_col.text_input(
            "Answer column",
            value=a_col_default,
            placeholder="e.g. answer  or  1",
            key=f"{key_prefix}_csv_a_col",
            help="Column header name or 0-based index for the answer field.",
        )
        csv_pairs_per_chunk = int(st.number_input(
            "Q&A pairs per chunk",
            min_value=1,
            max_value=20,
            value=pairs_default,
            step=1,
            key=f"{key_prefix}_csv_pairs",
            help=(
                "How many Q+A rows to group into one chunk. "
                "1 = maximum retrieval precision (recommended). "
                "3–5 = more context per chunk, useful when answers are very short."
            ),
        ))
        csv_format = "qa_pairs"
        csv_text_col = "0"

    elif chunking_strategy == "single_col":
        st.markdown("**CSV Format** *(single-column text)*")
        t_col_default = str(data_cfg.get("text_column", "0"))
        csv_text_col = st.text_input(
            "Text column",
            value=t_col_default,
            placeholder="e.g. text  or  0",
            key=f"{key_prefix}_csv_text_col",
            help="Column containing the text to index. Use header name or 0-based index.",
        )
        csv_format = "single_col"
        csv_question_col = "0"
        csv_answer_col = "1"
        csv_pairs_per_chunk = 1

    else:
        # Not CSV-specific — keep whatever was in the config
        csv_format = csv_format_default if csv_format_default != "raw" else "raw"
        csv_question_col = str(data_cfg.get("question_column", "0"))
        csv_answer_col = str(data_cfg.get("answer_column", "1"))
        csv_text_col = str(data_cfg.get("text_column", "0"))
        csv_pairs_per_chunk = int(data_cfg.get("pairs_per_chunk", 1))

    # Live chunk preview — shows real data from the user's files
    _render_chunk_preview(
        data_dir=cfg.get("paths", {}).get("data_dir", "data"),
        chunking_strategy=chunking_strategy,
        q_col=csv_question_col,
        a_col=csv_answer_col,
    )

    st.divider()

    # ── Section 2: Embedding ──────────────────────────────────────────────────
    st.markdown("**Embedding**")
    provider_options = ["sentence-transformers", "openai"]
    provider_default = str(cfg.get("embedding", {}).get("provider", "sentence-transformers"))
    embedding_provider = st.selectbox(
        "Provider",
        options=provider_options,
        format_func=lambda p: "openai — coming soon" if p == "openai" else p,
        index=provider_options.index(provider_default) if provider_default in provider_options else 0,
        key=f"{key_prefix}_embedding_provider",
        help="sentence-transformers runs locally for free. OpenAI support is coming soon.",
    )

    if embedding_provider == "openai":
        st.warning("OpenAI embeddings are coming soon. This is a free, local-first service — please use sentence-transformers.")

    # Preset models + custom option
    _PRESET_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "multi-qa-MiniLM-L6-cos-v1",
        "Custom...",
    ]
    _PRESET_DESCRIPTIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": "384-dim · Fast · Good general quality · Default",
        "BAAI/bge-small-en-v1.5":                 "384-dim · Same speed as MiniLM · Better retrieval quality (MTEB top performer)",
        "multi-qa-MiniLM-L6-cos-v1":              "384-dim · Fast · Trained on Q&A pairs — best for RAG retrieval",
        "Custom...":                               "Enter any sentence-transformers model from huggingface.co/models",
    }

    cfg_model = str(
        cfg.get("embedding", {}).get("sentence_transformers", {}).get(
            "model_name",
            cfg.get("embedding", {}).get("model", _PRESET_MODELS[0]),
        )
    )
    preset_default = cfg_model if cfg_model in _PRESET_MODELS else "Custom..."
    selected_preset = st.selectbox(
        "Model",
        options=_PRESET_MODELS,
        index=_PRESET_MODELS.index(preset_default),
        key=f"{key_prefix}_model_preset",
        help="Choose a preset model or select Custom to enter any Hugging Face model name.",
    )
    st.caption(_PRESET_DESCRIPTIONS.get(selected_preset, ""))

    if selected_preset == "Custom...":
        custom_default = cfg_model if cfg_model not in _PRESET_MODELS else ""
        embedding_model = st.text_input(
            "Custom model name",
            value=custom_default,
            placeholder="e.g. BAAI/bge-base-en-v1.5",
            key=f"{key_prefix}_embedding_model",
            help="Any model ID from huggingface.co/models that is compatible with sentence-transformers.",
        )
        if not embedding_model.strip():
            st.warning("Enter a model name to continue.")
    else:
        embedding_model = selected_preset
        # hidden input to keep session state key consistent
        st.session_state[f"{key_prefix}_embedding_model"] = embedding_model

    st.divider()

    # ── Section 3: Retrieval ──────────────────────────────────────────────────
    st.markdown("**Retrieval**")
    retrieval_cfg = cfg.get("retrieval", {})

    top_k = int(
        st.number_input(
            "Top K — chunks returned to LLM",
            min_value=1,
            value=int(retrieval_cfg.get("top_k", 4)),
            step=1,
            key=f"{key_prefix}_top_k",
            help="How many chunks are passed to generation. Higher = more context but slower and noisier. Typically 3–6.",
        )
    )

    st.markdown("**Re-ranking** *(optional — improves MRR)*")
    st.caption(
        "Re-ranking runs a second, more accurate model on the top candidates after the "
        "initial vector search. It improves ranking precision (MRR) at the cost of "
        "~50–150ms extra latency. Best free model: `BAAI/bge-reranker-base`."
    )

    _RERANKER_PRESETS = [
        "None — disabled",
        "cross-encoder/ms-marco-MiniLM-L-6-v2  (22MB · fast)",
        "cross-encoder/ms-marco-MiniLM-L-12-v2  (66MB · better quality)",
        "BAAI/bge-reranker-base  (110MB · best free option)",
        "Custom...",
    ]
    _RERANKER_MODEL_MAP = {
        "None — disabled": None,
        "cross-encoder/ms-marco-MiniLM-L-6-v2  (22MB · fast)": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2  (66MB · better quality)": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "BAAI/bge-reranker-base  (110MB · best free option)": "BAAI/bge-reranker-base",
    }

    cfg_reranker = retrieval_cfg.get("reranker_model") or None
    if cfg_reranker in _RERANKER_MODEL_MAP.values() and cfg_reranker is not None:
        reranker_preset_default = next(k for k, v in _RERANKER_MODEL_MAP.items() if v == cfg_reranker)
    else:
        reranker_preset_default = "None — disabled"

    reranker_preset = st.selectbox(
        "Re-ranker model",
        options=_RERANKER_PRESETS,
        index=_RERANKER_PRESETS.index(reranker_preset_default),
        key=f"{key_prefix}_reranker_preset",
        help="Cross-encoder model to re-score retrieved candidates. Downloads on first use.",
    )

    if reranker_preset == "Custom...":
        reranker_custom = st.text_input(
            "Custom re-ranker model name",
            placeholder="e.g. BAAI/bge-reranker-large",
            key=f"{key_prefix}_reranker_custom",
        )
        reranker_model = reranker_custom.strip() or None
    else:
        reranker_model = _RERANKER_MODEL_MAP.get(reranker_preset)

    if reranker_model:
        rerank_candidates = int(st.number_input(
            "Candidate pool size",
            min_value=top_k,
            max_value=100,
            value=int(retrieval_cfg.get("rerank_candidates", 20)),
            step=5,
            key=f"{key_prefix}_rerank_candidates",
            help=(
                f"Bi-encoder fetches this many candidates, then the cross-encoder re-scores them "
                f"and keeps top {top_k}. More candidates = better recall ceiling, slower re-ranking. "
                "20 is a good default."
            ),
        ))
        st.caption(
            f"Pipeline: vector search → top {rerank_candidates} candidates "
            f"→ cross-encoder re-score → top {top_k} passed to LLM"
        )
    else:
        rerank_candidates = int(retrieval_cfg.get("rerank_candidates", 20))

    return {
        "chunking_strategy": chunking_strategy,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "chunk_size_words": chunk_size,
        "chunk_overlap_words": chunk_overlap,
        "top_k": top_k,
        "reranker_model": reranker_model,
        "rerank_candidates": rerank_candidates,
        # CSV / data format settings
        "csv_format": csv_format,
        "csv_question_col": csv_question_col,
        "csv_answer_col": csv_answer_col,
        "csv_text_col": csv_text_col,
        "csv_pairs_per_chunk": csv_pairs_per_chunk,
    }


def _write_temp_config(base_cfg: dict, overrides: dict, tag: str,
                        provider_override: str | None = None) -> str:
    cfg = dict(base_cfg)
    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["chunking"] = dict(cfg.get("chunking", {}))
    cfg["embedding"] = dict(cfg.get("embedding", {}))
    cfg["retrieval"] = dict(cfg.get("retrieval", {}))
    cfg["embedding"]["sentence_transformers"] = dict(cfg["embedding"].get("sentence_transformers", {}))
    cfg["embedding"]["openai"] = dict(cfg["embedding"].get("openai", {}))

    tmp_index_dir = tempfile.mkdtemp(prefix=f"rag_index_{tag}_")
    tmp_raw_dir = tempfile.mkdtemp(prefix=f"rag_raw_{tag}_")
    cfg["paths"]["index_dir"] = tmp_index_dir
    cfg["paths"]["raw_docs_dir"] = tmp_raw_dir

    cfg["chunking"]["strategy"] = overrides["chunking_strategy"]
    cfg["chunking"]["chunk_size_words"] = int(overrides["chunk_size_words"])
    cfg["chunking"]["chunk_overlap_words"] = min(
        int(overrides["chunk_overlap_words"]),
        max(0, int(overrides["chunk_size_words"]) - 1),
    )
    cfg["embedding"]["provider"] = overrides["embedding_provider"]
    cfg["embedding"]["model"] = overrides["embedding_model"]
    if overrides["embedding_provider"] == "sentence-transformers":
        cfg["embedding"]["sentence_transformers"]["model_name"] = overrides["embedding_model"]
    if overrides["embedding_provider"] == "openai":
        cfg["embedding"]["openai"]["model"] = overrides["embedding_model"]
    cfg["retrieval"]["top_k"] = int(overrides["top_k"])
    cfg["retrieval"]["reranker_model"] = overrides.get("reranker_model") or None
    cfg["retrieval"]["rerank_candidates"] = int(overrides.get("rerank_candidates", 20))

    # Objective-driven provider override — takes precedence over config file
    if provider_override:
        cfg["generation"] = dict(cfg.get("generation", {}))
        cfg["generation"]["provider"] = provider_override

    # CSV / data format overrides
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["csv_format"] = overrides.get("csv_format", "raw")
    cfg["data"]["question_column"] = overrides.get("csv_question_col", "0")
    cfg["data"]["answer_column"] = overrides.get("csv_answer_col", "1")
    cfg["data"]["text_column"] = overrides.get("csv_text_col", "0")
    cfg["data"]["pairs_per_chunk"] = int(overrides.get("csv_pairs_per_chunk", 1))

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{tag}.yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp_path = tmp.name
    tmp.close()
    return tmp_path


def _validate_embedding_setup(config_path: str) -> None:
    cfg = load_config(config_path)
    embedding_cfg = cfg.get("embedding", {})
    provider = embedding_cfg.get("provider", "sentence-transformers")
    if provider != "openai":
        return

    openai_cfg = embedding_cfg.get("openai", {})
    api_key_env = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
    if not os.getenv(api_key_env):
        raise RuntimeError(
            f"OpenAI embedding selected but env var `{api_key_env}` is not set. "
            "Export it before running the evaluation."
        )


def _run_eval_with_optional_progress(
    config_path: str, dataset_path: str, status_slot, max_queries: int | None = None
) -> dict:
    _validate_embedding_setup(config_path)
    return run_eval(
        config_path=config_path,
        dataset_path=dataset_path,
        max_queries=max_queries,
        progress_callback=lambda msg: status_slot.info(msg),
    )


def _run_ab_with_optional_progress(
    config_a: str, config_b: str, dataset_path: str, status_slot, max_queries: int | None = None
) -> dict:
    _validate_embedding_setup(config_a)
    _validate_embedding_setup(config_b)
    return run_ab(
        config_a=config_a,
        config_b=config_b,
        dataset_path=dataset_path,
        metric_key=st.session_state.get("ab_metric_key", "recall_at_k"),
        max_queries=max_queries,
        progress_callback=lambda msg: status_slot.info(msg),
    )


def _metric_verdicts(summary: dict) -> list[tuple[str, str]]:
    verdicts: list[tuple[str, str]] = []
    recall = float(summary.get("avg_recall_at_k", 0.0))
    # Prefer BERTScore for the groundedness verdict — it's semantic, not token overlap
    bert = float(summary.get("avg_faithfulness_bertscore", 0.0))
    faith = bert if bert > 0.0 else float(summary.get("avg_faithfulness_proxy", 0.0))
    latency = float(summary.get("avg_latency_ms", 0.0))

    verdicts.append(("Retrieval Coverage", "Good" if recall >= 0.75 else "Needs Improvement"))
    verdicts.append(("Grounded Answers", "Good" if faith >= 0.80 else "Needs Improvement"))
    verdicts.append(("Speed", "Good" if latency <= 1200 else "Needs Improvement"))
    return verdicts


def _to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    fieldnames = sorted({k for row in rows for k in row.keys()})
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        flat = {k: (json.dumps(v, ensure_ascii=True) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        writer.writerow(flat)
    return output.getvalue().encode("utf-8")


def _render_index_info(index_dir: str, current_params: dict) -> None:
    """Show a sanity-check card: what model/dimensions the current index was built with,
    and whether it matches the selected embedding model."""
    meta_path = Path(index_dir) / "meta.json"
    if not meta_path.exists():
        st.caption("No index built yet — run an eval to build one.")
        return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return

    saved_model = meta.get("model_name", "unknown")
    current_model = current_params.get("embedding_model", "")
    match = saved_model == current_model

    with st.expander("Current Index Info", expanded=not match):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Built with model", saved_model)
        c2.metric("Dimensions", meta.get("dimensions", "—"))
        c3.metric("Chunks indexed", meta.get("num_chunks", "—"))
        c4.metric("Built at", meta.get("built_at", "—"))
        if not match and current_model:
            st.warning(
                f"Model mismatch: index uses `{saved_model}` but you selected `{current_model}`. "
                "The eval will automatically rebuild the index with the correct model."
            )
        elif match:
            st.success(f"Index matches selected model `{current_model}`.")


def _run_query_with_config(config_path: str, query_text: str) -> dict:
    pipeline = RAGPipeline(config_path=config_path)
    result = pipeline.query(query_text)
    chunk_map = {c["chunk_id"]: c for c in pipeline.index.chunks}
    enriched_sources = []
    for src in result.get("sources", []):
        chunk = chunk_map.get(src.get("chunk_id"))
        enriched_sources.append(
            {
                "doc_id": src.get("doc_id"),
                "chunk_id": src.get("chunk_id"),
                "score": src.get("score"),
                "preview": (chunk.get("text", "")[:240] + "...") if chunk and chunk.get("text") else "",
            }
        )
    result["sources"] = enriched_sources
    return result


def render_single(report: dict) -> None:
    with st.expander("Metric Guide", expanded=False):
        st.markdown(
            "| Metric | What it measures | Reliable with Simple generation? |\n"
            "|---|---|---|\n"
            "| **Recall@k** | Did we retrieve the right document? | ✅ Yes |\n"
            "| **MRR** | How high did the right document rank? | ✅ Yes |\n"
            "| **Faithfulness (token)** | Word overlap between answer and context | ⚠️ Inflated — answer IS the chunks |\n"
            "| **Faithfulness (BERTScore)** | Semantic similarity of answer to context using BERT embeddings | ⚠️ Inflated for same reason |\n"
            "| **Latency** | Avg ms per query | ✅ Yes |\n"
            "| **Confidence** | Heuristic from retrieval scores | ✅ Yes |\n"
            "| **Hallucination Rate** | Fraction of queries flagged by either Type 1 or Type 2 | ✅ Yes |\n"
            "| **Type 1 Hallucination** | Retrieval failure — context not relevant to query (low confidence) | ✅ Yes |\n"
            "| **Type 2 Hallucination** | Generation unfaithfulness — answer not supported by retrieved context (NLI) | ✅ Yes (needs LLM gen) |"
        )

    summary = report["summary"]
    generation_provider = summary.get("generation_provider", "simple")

    if generation_provider == "simple":
        st.info(
            "ℹ️ **Generation provider is `simple`** — the answer is just the retrieved chunks "
            "concatenated together. Both faithfulness metrics are inflated and not meaningful. "
            "**Use Recall@k and MRR as your primary signals.** "
            "Switch to `ollama` or `openai` generation in `configs/base.yaml` for real faithfulness scores."
        )

    col1, col2 = st.columns(2)
    col1.metric("Avg Recall@k", f"{summary['avg_recall_at_k']:.3f}")
    col2.metric("Avg MRR", f"{summary['avg_mrr']:.3f}")

    col3, col4 = st.columns(2)
    col3.metric(
        "Faithfulness — Token Overlap",
        f"{summary['avg_faithfulness_proxy']:.3f}",
        help="Fraction of answer words that appear in retrieved chunks. Fast but unreliable — inflated when generation=simple.",
    )

    bert_fallback = summary.get("bertscore_fallback", False)
    bert_score_val = summary.get("avg_faithfulness_bertscore", 0.0)
    proxy_val = summary.get("avg_faithfulness_proxy", 0.0)

    if bert_fallback:
        bert_fallback_reason = summary.get("bertscore_fallback_reason", "unknown error")
        col4.metric(
            "Faithfulness — BERTScore ⚠️",
            f"{bert_score_val:.3f}",
            help=f"BERTScore fell back to token overlap — values will match the Token Overlap column. Reason: {bert_fallback_reason}",
        )
        st.warning(
            f"⚠️ **BERTScore fell back to token overlap** — both faithfulness numbers are showing the same value. "
            f"Reason: `{bert_fallback_reason}`. "
            "Run `pip install bert-score` in your virtual environment and re-run the eval to get real BERTScore values."
        )
    else:
        col4.metric(
            "Faithfulness — BERTScore",
            f"{bert_score_val:.3f}",
            help="Semantic similarity between answer and context using BERT embeddings. More meaningful than token overlap, but still inflated when generation=simple.",
        )

    col5, col6, col7 = st.columns(3)
    col5.metric("Latency (ms)", f"{summary['avg_latency_ms']:.2f}")
    col6.metric("Confidence", f"{summary['avg_confidence']:.3f}")
    col7.metric("Hallucination Rate", f"{summary['hallucination_rate']:.3f}")

    # ── Hallucination Breakdown ───────────────────────────────────────────────
    st.subheader("Hallucination Breakdown")
    st.caption(
        "**Type 1** — Retrieval failure: the retrieved chunks were not relevant (low confidence score). "
        "**Type 2** — Generation unfaithfulness: the LLM made claims not supported by the context (NLI check). "
        "Type 2 is automatically skipped for `simple` generation (answer = chunks, NLI would always return 1.0)."
    )

    # Type 2 is None when all queries were skipped (e.g. simple generation)
    type2_ran_count = summary.get("type2_nli_ran_count", 0)
    type2_skipped_count = summary.get("type2_nli_skipped_count", 0)
    type2_was_skipped = summary.get("type2_hallucination_rate") is None

    h_col1, h_col2, h_col3 = st.columns(3)

    type1_rate = summary.get("type1_hallucination_rate", summary.get("hallucination_rate", 0.0))
    type2_rate = summary.get("type2_hallucination_rate")   # None when skipped
    type2_score = summary.get("avg_type2_nli_score")       # None when skipped

    if type1_rate >= 0.3:
        h_col1.error(f"🔴 Type 1 Rate: {type1_rate:.1%}")
    elif type1_rate >= 0.1:
        h_col1.warning(f"🟡 Type 1 Rate: {type1_rate:.1%}")
    else:
        h_col1.success(f"🟢 Type 1 Rate: {type1_rate:.1%}")
    h_col1.caption("Retrieval failure (low confidence)")

    if type2_was_skipped:
        h_col2.info("⏭️ Type 2: Skipped")
        h_col2.caption(
            f"Use ollama generation to enable NLI check. "
            f"{type2_skipped_count} queries skipped."
        )
        h_col3.info("⏭️ NLI Score: N/A")
        h_col3.caption("Not computed — switch to ollama to measure")
    else:
        if type2_rate >= 0.3:
            h_col2.error(f"🔴 Type 2 Rate: {type2_rate:.1%}")
        elif type2_rate >= 0.1:
            h_col2.warning(f"🟡 Type 2 Rate: {type2_rate:.1%}")
        else:
            h_col2.success(f"🟢 Type 2 Rate: {type2_rate:.1%}")
        h_col2.caption(f"Generation unfaithfulness (NLI) — {type2_ran_count} queries measured")

        if type2_score >= 0.8:
            h_col3.success(f"🟢 NLI Score: {type2_score:.3f}")
        elif type2_score >= 0.5:
            h_col3.warning(f"🟡 NLI Score: {type2_score:.3f}")
        else:
            h_col3.error(f"🔴 NLI Score: {type2_score:.3f}")
        h_col3.caption("Avg sentence entailment (1.0 = fully grounded)")

    st.subheader("PM Verdict")
    verdict_cols = st.columns(3)
    for idx, (label, verdict) in enumerate(_metric_verdicts(summary)):
        if verdict == "Good":
            verdict_cols[idx].success(f"{label}: {verdict}")
        else:
            verdict_cols[idx].warning(f"{label}: {verdict}")

    st.subheader("Query-Level Metrics")
    st.dataframe(report["queries"], use_container_width=True)
    st.download_button(
        "Download Query Metrics CSV",
        data=_to_csv_bytes(report["queries"]),
        file_name="single_eval_query_metrics.csv",
        mime="text/csv",
    )


def render_ab(report_a: dict, report_b: dict) -> None:
    s1 = report_a["summary"]
    s2 = report_b["summary"]

    gen_a = s1.get("generation_provider", "simple")
    gen_b = s2.get("generation_provider", "simple")
    if gen_a == "simple" or gen_b == "simple":
        st.info(
            "ℹ️ **Generation provider is `simple`** — faithfulness metrics are inflated. "
            "Focus on **Recall@k** and **MRR** for this comparison."
        )

    st.subheader("A/B Summary")
    compare_rows = [
        {
            "metric": "avg_recall_at_k",
            "Control": s1["avg_recall_at_k"],
            "Test": s2["avg_recall_at_k"],
            "delta_test_minus_control": s2["avg_recall_at_k"] - s1["avg_recall_at_k"],
        },
        {
            "metric": "avg_mrr",
            "Control": s1["avg_mrr"],
            "Test": s2["avg_mrr"],
            "delta_test_minus_control": s2["avg_mrr"] - s1["avg_mrr"],
        },
        {
            "metric": "avg_faithfulness_bertscore",
            "Control": s1.get("avg_faithfulness_bertscore", 0.0),
            "Test": s2.get("avg_faithfulness_bertscore", 0.0),
            "delta_test_minus_control": s2.get("avg_faithfulness_bertscore", 0.0) - s1.get("avg_faithfulness_bertscore", 0.0),
        },
        {
            "metric": "avg_faithfulness_proxy (token overlap)",
            "Control": s1["avg_faithfulness_proxy"],
            "Test": s2["avg_faithfulness_proxy"],
            "delta_test_minus_control": s2["avg_faithfulness_proxy"] - s1["avg_faithfulness_proxy"],
        },
        {
            "metric": "avg_latency_ms",
            "Control": s1["avg_latency_ms"],
            "Test": s2["avg_latency_ms"],
            "delta_test_minus_control": s2["avg_latency_ms"] - s1["avg_latency_ms"],
        },
        {
            "metric": "type1_hallucination_rate (retrieval failure ↓ better)",
            "Control": s1.get("type1_hallucination_rate", s1.get("hallucination_rate", "—")),
            "Test": s2.get("type1_hallucination_rate", s2.get("hallucination_rate", "—")),
            "delta_test_minus_control": (
                round(
                    s2.get("type1_hallucination_rate", s2.get("hallucination_rate", 0.0) or 0.0)
                    - s1.get("type1_hallucination_rate", s1.get("hallucination_rate", 0.0) or 0.0),
                    4,
                )
                if s1.get("type1_hallucination_rate") is not None and s2.get("type1_hallucination_rate") is not None
                else "—"
            ),
        },
        {
            "metric": "type2_hallucination_rate (NLI unfaithfulness ↓ better — ollama only)",
            "Control": f"{s1['type2_hallucination_rate']:.1%}" if s1.get("type2_hallucination_rate") is not None else "⏭️ skipped",
            "Test": f"{s2['type2_hallucination_rate']:.1%}" if s2.get("type2_hallucination_rate") is not None else "⏭️ skipped",
            "delta_test_minus_control": (
                round(s2["type2_hallucination_rate"] - s1["type2_hallucination_rate"], 4)
                if s1.get("type2_hallucination_rate") is not None and s2.get("type2_hallucination_rate") is not None
                else "—"
            ),
        },
        {
            "metric": "avg_type2_nli_score (grounding ↑ better — ollama only)",
            "Control": f"{s1['avg_type2_nli_score']:.3f}" if s1.get("avg_type2_nli_score") is not None else "⏭️ skipped",
            "Test": f"{s2['avg_type2_nli_score']:.3f}" if s2.get("avg_type2_nli_score") is not None else "⏭️ skipped",
            "delta_test_minus_control": (
                round(s2["avg_type2_nli_score"] - s1["avg_type2_nli_score"], 4)
                if s1.get("avg_type2_nli_score") is not None and s2.get("avg_type2_nli_score") is not None
                else "—"
            ),
        },
    ]

    # BERTScore fallback warning for A/B
    bert_fb_a = s1.get("bertscore_fallback", False)
    bert_fb_b = s2.get("bertscore_fallback", False)
    if bert_fb_a or bert_fb_b:
        which = "both configs" if (bert_fb_a and bert_fb_b) else ("Control" if bert_fb_a else "Test")
        st.warning(
            f"⚠️ **BERTScore fell back to token overlap for {which}** — the faithfulness_bertscore row "
            "shows token overlap, not semantic BERT similarity. "
            "Run `pip install bert-score` and re-run the eval to fix this."
        )

    st.dataframe(compare_rows, use_container_width=True)

    st.subheader("Per-Query Retrieval Trace")
    q1 = report_a.get("queries", [])
    q2 = report_b.get("queries", [])
    rows = []
    for i in range(min(len(q1), len(q2))):
        control_trace = q1[i].get("retrieval_trace", [])
        test_trace = q2[i].get("retrieval_trace", [])
        rows.append(
            {
                "query": q1[i].get("query"),
                "control_top_chunks": [x.get("chunk_id") for x in control_trace],
                "test_top_chunks": [x.get("chunk_id") for x in test_trace],
                "control_preview": " | ".join([x.get("preview", "") for x in control_trace[:2]]),
                "test_preview": " | ".join([x.get("preview", "") for x in test_trace[:2]]),
            }
        )
    st.dataframe(rows, use_container_width=True)
    st.download_button(
        "Download A/B Trace CSV",
        data=_to_csv_bytes(rows),
        file_name="ab_retrieval_trace.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="RAG Eval Dashboard", layout="wide")
st.title("RAG Evaluation Dashboard")

report_files = sorted(glob.glob("eval/reports/eval_*.json"), reverse=True)
dataset_files = sorted(glob.glob("eval/datasets/*.jsonl"))

st.markdown("### Step 1: Upload Documents")

with st.expander("📋 What file formats are supported?", expanded=False):
    st.markdown(
        "| Format | Works out of the box | Notes |\n"
        "|---|---|---|\n"
        "| `.txt` `.md` | ✅ Yes | Best for articles, wikis, documentation |\n"
        "| `.csv` (Q&A columns) | ✅ Yes | Use **qa_pair** strategy — each row becomes one chunk |\n"
        "| `.csv` (single text column) | ✅ Yes | Use **single_col** mode in CSV Format settings |\n"
        "| `.csv` (raw / mixed) | ✅ Yes | Use **fixed** strategy — treated as plain text |\n"
        "| `.pdf` | ✅ Yes | Requires `pip install pypdf` |\n"
        "| `.json` | ✅ Yes | Treated as formatted text — best for structured data dumps |\n"
        "\n"
        "**⚠️ CSV tip:** If your CSV has a `question` and `answer` column, "
        "don't use `fixed` chunking — it will randomly cut across rows. "
        "Select **qa_pair** in the Chunking Strategy dropdown below."
    )

uploads = st.file_uploader(
    "Upload your files for indexing",
    type=["txt", "md", "csv", "json", "pdf"],
    accept_multiple_files=True,
)
if st.button("Save Uploaded Files"):
    if uploads:
        saved_files = _save_uploaded_files(uploads, data_dir="data")
        st.success(f"✅ Saved {len(saved_files)} file(s) to `data/`: {', '.join(saved_files)}")

        # After save: show what was detected and what to do next
        info = _scan_data_folder("data")
        strategy, explanation = _recommend_strategy(info["counts"])
        st.info(
            f"💡 **Next step:** Go to **Step 3 → Chunking Strategy** and select `{strategy}`. "
            f"{explanation}"
        )
    else:
        st.info("No files selected.")

# Always show current data folder status
_render_data_advisor("data")

st.markdown("### Step 2: Build Eval Dataset")
if "eval_builder_rows" not in st.session_state:
    st.session_state["eval_builder_rows"] = []
data_files = _list_data_files()
builder_col1, builder_col2 = st.columns(2)
query_input = builder_col1.text_input("Question", key="builder_query")
ref_answer_input = builder_col2.text_input("Reference Answer", key="builder_reference")
doc_id_input = st.selectbox(
    "Expected Document ID",
    options=data_files if data_files else ["(no files in data/)"],
    index=0,
)
add_col1, add_col2 = st.columns(2)
if add_col1.button("Add Eval Row"):
    if query_input.strip() and ref_answer_input.strip() and doc_id_input != "(no files in data/)":
        st.session_state["eval_builder_rows"].append(
            {
                "query": query_input.strip(),
                "reference_answer": ref_answer_input.strip(),
                "expected_doc_ids": [doc_id_input],
            }
        )
        st.success("Row added.")
    else:
        st.warning("Question, Reference Answer, and Expected Document are required.")
if add_col2.button("Clear Eval Rows"):
    st.session_state["eval_builder_rows"] = []

rows = st.session_state["eval_builder_rows"]
if rows:
    st.dataframe(rows, use_container_width=True)
dataset_name = st.text_input("Dataset filename", value="pm_eval.jsonl")
if st.button("Save Eval Dataset"):
    if not rows:
        st.warning("No eval rows to save.")
    else:
        output_path = _save_eval_dataset(rows, f"eval/datasets/{dataset_name}")
        st.success(f"Saved dataset to {output_path}")
        dataset_files = sorted(glob.glob("eval/datasets/*.jsonl"))
else:
    st.info("No eval rows yet. Add at least one question.")

mode = st.radio("Mode", ["Single Run", "A/B Compare"], horizontal=True)

if mode == "Single Run":
    if not dataset_files:
        st.error("Missing dataset files. Expected `eval/datasets/*.jsonl`.")
        st.stop()
    base_config = "configs/base.yaml"
    if not Path(base_config).exists():
        st.error("Missing base config `configs/base.yaml`.")
        st.stop()

    st.subheader("Step 3: Run Single Evaluation")
    cfg = _read_cfg(base_config)

    # ── Objective selector — drives provider + dataset + max_queries ──────────
    st.divider()
    obj_settings = _render_objective_selector("single", dataset_files)
    dataset = obj_settings["selected_dataset"]
    single_max_q = obj_settings["max_queries"]
    provider_override = obj_settings["provider_override"]
    st.divider()

    with st.expander("⚙️ Advanced — Chunking, Embedding & Retrieval Settings", expanded=False):
        with st.expander("Parameter Guide — what each setting does per strategy", expanded=False):
            st.markdown(_PARAM_GUIDE_TABLE)

    # _config_editor must always run (Streamlit requires consistent widget order)
    single_params = _config_editor("Parameters", cfg, "single")
    _render_index_info(cfg["paths"].get("index_dir", "artifacts/index"), single_params)
    if single_params["chunking_strategy"] == "qa_pair":
        st.info(
            "ℹ️ **qa_pair strategy** — each CSV row becomes one chunk. "
            "Column names set above will apply automatically."
        )

    if st.button("▶️ Run Eval", type="primary"):
        if single_params["chunking_strategy"] == "semantic":
            st.error("Semantic chunking is coming soon. Please select a different strategy.")
            st.stop()
        if single_params["embedding_provider"] == "openai":
            st.error("OpenAI embeddings are coming soon. Please use sentence-transformers.")
            st.stop()
        status = st.empty()
        status.info(f"Starting eval — objective: {obj_settings['objective_name']}, provider: `{provider_override}`")
        try:
            temp_config = _write_temp_config(cfg, single_params, "single",
                                              provider_override=provider_override)
            report = _run_eval_with_optional_progress(
                config_path=temp_config,
                dataset_path=dataset,
                status_slot=status,
                max_queries=int(single_max_q),
            )
            report["effective_config"] = temp_config
            report["objective"] = obj_settings["objective_name"]
            st.session_state["single_report"] = report
            status.success(
                f"✅ Eval complete — {report['summary']['num_queries']} queries · "
                f"provider: `{provider_override}` · "
                f"focus: `{'`, `'.join(obj_settings['focus_metrics'])}`"
            )
        except Exception as exc:
            status.error(f"Evaluation failed: {exc}")
            st.exception(exc)

    st.subheader("Step 3: Query Playground")
    query_text = st.text_area("Ask a question against current single-run settings", height=100)
    if st.button("Run Query"):
        if not query_text.strip():
            st.warning("Please enter a question.")
        else:
            status_q = st.empty()
            status_q.info("Running query...")
            try:
                temp_config = _write_temp_config(cfg, single_params, "single_query")
                query_result = _run_query_with_config(temp_config, query_text.strip())
                status_q.success("Query complete")
                st.write("**Answer**")
                st.write(query_result.get("answer", ""))
                qc1, qc2, qc3, qc4 = st.columns(4)
                qc1.metric("Confidence", f"{query_result.get('confidence', 0.0):.3f}")
                qc2.metric("Hallucination Risk", "Yes ⚠️" if query_result.get("hallucination_risk") else "No ✅")

                hallucination = query_result.get("hallucination", {})
                type1 = hallucination.get("type1", {})
                type2 = hallucination.get("type2", {})

                # Type 1 — retrieval confidence
                t1_risk = type1.get("risk_level", "unknown")
                t1_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(t1_risk, "⚪")
                qc3.metric("Type 1 (retrieval)", f"{t1_color} {t1_risk}")

                # Type 2 — NLI grounding score
                t2_score = type2.get("score", None)
                if t2_score is not None:
                    t2_label = f"{'🔴' if t2_score < 0.5 else '🟡' if t2_score < 0.8 else '🟢'} {t2_score:.2f}"
                    t2_caption = "NLI grounding score"
                else:
                    t2_label = "—"
                    t2_caption = "NLI skipped"
                qc4.metric("Type 2 (NLI grounding)", t2_label)

                if type1.get("reason"):
                    st.caption(f"Type 1 reason: {type1['reason']}")
                if type2.get("flagged"):
                    unfaithful = [d["sentence"] for d in type2.get("detail", []) if d["label"] != "entailment"]
                    if unfaithful:
                        with st.expander("Type 2 — Ungrounded sentences", expanded=False):
                            for s in unfaithful:
                                st.markdown(f"- _{s}_")

                st.write("**Retrieved Context**")
                st.dataframe(query_result.get("sources", []), use_container_width=True)
            except Exception as exc:
                status_q.error(f"Query failed: {exc}")
                st.exception(exc)

    single_report = st.session_state.get("single_report")
    if single_report:
        obj_name = single_report.get("objective", "")
        if obj_name and obj_name in _OBJECTIVES:
            obj_meta = _OBJECTIVES[obj_name]
            st.info(
                f"**Results for objective: {obj_name}** — "
                f"Focus on: `{'`  ·  `'.join(obj_meta['focus_metrics'])}`"
            )
        st.caption(f"Effective config: {single_report.get('effective_config')}")
        render_single(single_report)
    else:
        st.info("No in-session result yet — select an objective above and click ▶️ Run Eval.")
else:
    if not dataset_files:
        st.error("Missing dataset files. Expected `eval/datasets/*.jsonl`.")
        st.stop()
    base_config = "configs/base.yaml"
    if not Path(base_config).exists():
        st.error("Missing base config `configs/base.yaml`.")
        st.stop()

    st.subheader("Step 3: Run A/B Test")

    # ── Objective selector — drives provider + dataset + max_queries ──────────
    st.divider()
    ab_obj = _render_objective_selector("ab", dataset_files)
    dataset = ab_obj["selected_dataset"]
    ab_max_q = ab_obj["max_queries"]
    ab_provider_override = ab_obj["provider_override"]
    st.info(
        f"⚠️ A/B runs **twice** (control + test) = **{ab_max_q * 2} total queries**. "
        f"Provider auto-set to `{ab_provider_override}` based on your objective."
    )
    st.divider()

    metric_options = {
        "Recall@k — retrieval coverage": "recall_at_k",
        "MRR — ranking quality": "mrr",
        "Faithfulness — BERTScore (use with ollama)": "faithfulness_bertscore",
        "Faithfulness — token overlap": "faithfulness_proxy",
        "Latency — lower is better": "latency_ms",
        "Confidence": "confidence",
    }
    # Auto-suggest primary metric based on objective
    _obj_to_metric = {
        "🔍 Test Retrieval Quality": "Recall@k — retrieval coverage",
        "📊 Compare Chunking / Embedding": "Recall@k — retrieval coverage",
        "🛡️ Measure Hallucination": "Recall@k — retrieval coverage",
        "📝 Evaluate Answer Quality": "Faithfulness — BERTScore (use with ollama)",
        "✅ Full Validation": "Recall@k — retrieval coverage",
    }
    suggested_metric = _obj_to_metric.get(ab_obj["objective_name"], "Recall@k — retrieval coverage")
    metric_idx = list(metric_options.keys()).index(suggested_metric)
    metric_label = st.selectbox("Primary A/B Metric (for winner determination)",
                                 list(metric_options.keys()), index=metric_idx)
    st.session_state["ab_metric_key"] = metric_options[metric_label]

    with st.expander("⚙️ Advanced — Control vs Test Parameters", expanded=True):
        with st.expander("Parameter Guide", expanded=False):
            st.markdown(_PARAM_GUIDE_TABLE)
        cfg_a = _read_cfg(base_config)
        cfg_b = _read_cfg(base_config)
        left, right = st.columns(2)
        with left:
            overrides_a = _config_editor("Control Parameters", cfg_a, "ab_control")
        with right:
            overrides_b = _config_editor("Test Parameters", cfg_b, "ab_test")
        _render_index_info(cfg_a["paths"].get("index_dir", "artifacts/index"), overrides_a)

    if st.button("▶️ Run A/B", type="primary"):
        if overrides_a["chunking_strategy"] == "semantic" or overrides_b["chunking_strategy"] == "semantic":
            st.error("Semantic chunking is coming soon. Please select a different strategy.")
            st.stop()
        if overrides_a["embedding_provider"] == "openai" or overrides_b["embedding_provider"] == "openai":
            st.error("OpenAI embeddings are coming soon. Please use sentence-transformers.")
            st.stop()
        status = st.empty()
        status.info(
            f"Starting A/B — objective: {ab_obj['objective_name']} · "
            f"provider: `{ab_provider_override}` · {ab_max_q} queries × 2"
        )
        try:
            temp_config_a = _write_temp_config(cfg_a, overrides_a, "control",
                                                provider_override=ab_provider_override)
            temp_config_b = _write_temp_config(cfg_b, overrides_b, "test",
                                                provider_override=ab_provider_override)
            result = _run_ab_with_optional_progress(
                config_a=temp_config_a,
                config_b=temp_config_b,
                dataset_path=dataset,
                status_slot=status,
                max_queries=int(ab_max_q),
            )
            result["effective_control_config"] = temp_config_a
            result["effective_test_config"] = temp_config_b
            result["objective"] = ab_obj["objective_name"]
            st.session_state["ab_result"] = result
            status.success(
                f"✅ A/B complete — focus: `{'`, `'.join(ab_obj['focus_metrics'])}`"
            )
        except Exception as exc:
            status.error(f"A/B evaluation failed: {exc}")
            st.exception(exc)

    ab_result = st.session_state.get("ab_result")
    if ab_result:
        winner = ab_result["winner"]
        winner_label = "Control" if winner == "A" else "Test" if winner == "B" else "Inconclusive"
        metric_name = ab_result["metric"]
        delta = float(ab_result.get("mean_delta_b_minus_a", 0.0))
        p_val = float(ab_result.get("p_value", 1.0))
        confidence_phrase = "high statistical confidence" if p_val < 0.01 else "moderate confidence" if p_val < 0.05 else "low confidence"
        direction = "improves" if delta >= 0 else "reduces"
        plain_summary = (
            f"{winner_label} is the current winner on `{metric_name}` with {confidence_phrase} "
            f"(p={p_val:.4f}). Test {direction} the selected metric by {abs(delta):.4f} vs Control."
        )
        st.success(
            f"Winner: {winner_label} | "
            f"Metric: {ab_result['metric']} | "
            f"p-value: {ab_result['p_value']:.4f}"
        )
        st.info(plain_summary)
        st.caption(
            f"Effective configs: Control={ab_result.get('effective_control_config')} | "
            f"Test={ab_result.get('effective_test_config')}"
        )
        left_report = load_report(ab_result["report_path_a"])
        right_report = load_report(ab_result["report_path_b"])
        render_ab(left_report, right_report)
        st.download_button(
            "Download A/B Summary JSON",
            data=json.dumps(ab_result, ensure_ascii=True, indent=2).encode("utf-8"),
            file_name="ab_summary.json",
            mime="application/json",
        )
    else:
        st.info("No in-session A/B result yet. Click `Run A/B`.")
