from __future__ import annotations

import json
from pathlib import Path

from src.common.io import read_csv_as_docs, read_supported_file

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf"}


def load_documents(data_dir: str, data_config: dict | None = None) -> list[dict]:
    """Load all supported files from data_dir into a flat list of doc dicts.

    For CSV files the behaviour is controlled by data_config:
      csv_format=raw        — one doc per file (old behaviour, default)
      csv_format=qa_pairs   — one doc per Q+A row (or group of rows)
      csv_format=single_col — one doc per cell in the text column

    Non-CSV files always produce one doc per file.
    """
    cfg = data_config or {}
    csv_format = str(cfg.get("csv_format", "raw")).lower()
    question_column = str(cfg.get("question_column", "0"))
    answer_column = str(cfg.get("answer_column", "1"))
    text_column = str(cfg.get("text_column", "0"))
    pairs_per_chunk = int(cfg.get("pairs_per_chunk", 1))

    docs: list[dict] = []
    root = Path(data_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".csv" and csv_format != "raw":
            # Structured CSV ingestion — returns multiple docs (one per row/group)
            try:
                csv_docs = read_csv_as_docs(
                    path=path,
                    csv_format=csv_format,
                    question_column=question_column,
                    answer_column=answer_column,
                    text_column=text_column,
                    pairs_per_chunk=pairs_per_chunk,
                )
                docs.extend(csv_docs)
            except Exception as exc:
                # Surface as a clear error so the user can fix column config
                raise RuntimeError(
                    f"Failed to load CSV '{path.name}' with format '{csv_format}'. "
                    f"Check your data.csv_format settings. Error: {exc}"
                ) from exc
        else:
            # Plain text / non-CSV / raw CSV
            text = read_supported_file(path)
            docs.append(
                {
                    "doc_id": path.name,
                    "path": str(path),
                    "text": text,
                }
            )

    return docs


def persist_raw_docs(docs: list[dict], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "raw_docs.json"
    output_path.write_text(json.dumps(docs, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(output_path)
