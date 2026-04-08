from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _resolve_column(header: list[str], col_spec: str) -> int:
    """Resolve a column spec (name or 0-based int string) to an index.

    Examples:
        "question"  → index of "question" header
        "0"         → 0
        "2"         → 2
    """
    # Try integer index first
    try:
        idx = int(col_spec)
        if 0 <= idx < len(header):
            return idx
        raise ValueError(f"Column index {idx} out of range (file has {len(header)} columns)")
    except ValueError:
        pass

    # Try header name (case-insensitive)
    col_lower = col_spec.lower().strip()
    for i, h in enumerate(header):
        if h.lower().strip() == col_lower:
            return i

    raise ValueError(
        f"Column '{col_spec}' not found. Available columns: {header}"
    )


def read_supported_file(path: Path) -> str:
    """Read a file to a plain string.

    For CSVs this returns a naive pipe-joined blob — used only when
    csv_format=raw.  Use read_csv_as_docs() for structured ingestion.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".csv":
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        return "\n".join(rows)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=True, indent=2)

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PDF support requires `pypdf`.") from exc

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    raise ValueError(f"Unsupported file type: {path}")


def read_csv_as_docs(
    path: Path,
    csv_format: str = "raw",
    question_column: str = "0",
    answer_column: str = "1",
    text_column: str = "0",
    pairs_per_chunk: int = 1,
) -> list[dict]:
    """Parse a CSV into a list of doc dicts suitable for the ingestion pipeline.

    csv_format options
    ------------------
    raw
        One doc for the whole file (same as old behaviour — flat text blob).
    qa_pairs
        Two-column structured Q&A.  One doc per row (or per `pairs_per_chunk`
        rows when > 1), with text = "Q: {question}\\nA: {answer}".
    single_col
        One column of text.  One doc per non-empty cell.

    Returns a list of dicts: [{doc_id, path, text}, ...]
    The doc_id encodes the source file and row index so retrieval traces
    are still human-readable.
    """
    filename = path.name
    fmt = csv_format.lower().strip()

    if fmt == "raw":
        # Old behaviour — caller should use read_supported_file instead, but
        # we support it here for uniform calling convention.
        return [
            {
                "doc_id": filename,
                "path": str(path),
                "text": read_supported_file(path),
            }
        ]

    # ── Read the CSV ──────────────────────────────────────────────────────────
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        all_rows = [r for r in reader if any(cell.strip() for cell in r)]

    if not all_rows:
        return []

    # First row may or may not be a header — we detect by trying to resolve
    # the column spec.  If the spec is an integer, no header is needed.
    has_header = not question_column.lstrip("-").isdigit() or not answer_column.lstrip("-").isdigit()

    if has_header and all_rows:
        header = all_rows[0]
        data_rows = all_rows[1:]
    else:
        # Generate synthetic header so _resolve_column works uniformly
        header = [str(i) for i in range(len(all_rows[0]))]
        data_rows = all_rows

    # ── qa_pairs ──────────────────────────────────────────────────────────────
    if fmt == "qa_pairs":
        try:
            q_idx = _resolve_column(header, question_column)
            a_idx = _resolve_column(header, answer_column)
        except ValueError as exc:
            raise ValueError(
                f"CSV column config error in '{filename}': {exc}"
            ) from exc

        pairs: list[tuple[str, str]] = []
        for row in data_rows:
            q = row[q_idx].strip() if q_idx < len(row) else ""
            a = row[a_idx].strip() if a_idx < len(row) else ""
            if q or a:
                pairs.append((q, a))

        # Group pairs_per_chunk pairs into one chunk
        g = max(1, pairs_per_chunk)
        docs: list[dict] = []
        for start in range(0, len(pairs), g):
            group = pairs[start : start + g]
            if g == 1:
                q, a = group[0]
                text = f"Q: {q}\nA: {a}"
            else:
                lines = [f"Q: {q}\nA: {a}" for q, a in group]
                text = "\n\n".join(lines)
            row_label = f"rows_{start+1}-{start+len(group)}"
            docs.append(
                {
                    "doc_id": f"{filename}::{row_label}",
                    "path": str(path),
                    "text": text,
                }
            )
        return docs

    # ── single_col ────────────────────────────────────────────────────────────
    if fmt == "single_col":
        try:
            t_idx = _resolve_column(header, text_column)
        except ValueError as exc:
            raise ValueError(
                f"CSV column config error in '{filename}': {exc}"
            ) from exc

        docs = []
        for row_num, row in enumerate(data_rows, start=1):
            text = row[t_idx].strip() if t_idx < len(row) else ""
            if text:
                docs.append(
                    {
                        "doc_id": f"{filename}::row_{row_num}",
                        "path": str(path),
                        "text": text,
                    }
                )
        return docs

    raise ValueError(
        f"Unknown csv_format '{csv_format}'. Choose: raw | qa_pairs | single_col"
    )


def read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items
