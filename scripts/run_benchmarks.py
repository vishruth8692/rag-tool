"""Run all benchmark configurations and print a summary table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.runner import run_eval

DATASET = "eval/datasets/wikipedia_eval_hard.jsonl"
MAX_QUERIES = 200

BENCHMARKS = [
    {
        "label": "fixed  · all-MiniLM-L6-v2",
        "config": "configs/bench_fixed_minilm.yaml",
        "chunking": "fixed",
        "embedding": "all-MiniLM-L6-v2",
    },
    {
        "label": "paragraph · all-MiniLM-L6-v2",
        "config": "configs/bench_paragraph_minilm.yaml",
        "chunking": "paragraph",
        "embedding": "all-MiniLM-L6-v2",
    },
    {
        "label": "paragraph · bge-small-en-v1.5",
        "config": "configs/bench_paragraph_bge.yaml",
        "chunking": "paragraph",
        "embedding": "bge-small-en-v1.5",
    },
    {
        "label": "paragraph · multi-qa-MiniLM-L6-cos-v1",
        "config": "configs/bench_paragraph_multiqa.yaml",
        "chunking": "paragraph",
        "embedding": "multi-qa-MiniLM-L6-cos-v1",
    },
]

results = []
for b in BENCHMARKS:
    print(f"\n{'='*60}")
    print(f"Running: {b['label']}")
    print('='*60)
    try:
        report = run_eval(
            config_path=b["config"],
            dataset_path=DATASET,
            max_queries=MAX_QUERIES,
            progress_callback=lambda m: print(f"  {m}"),
        )
        s = report["summary"]
        results.append({
            **b,
            "recall": s["avg_recall_at_k"],
            "mrr": s["avg_mrr"],
            "latency_ms": s["avg_latency_ms"],
            "num_queries": s["num_queries"],
            "confidence": s["avg_confidence"],
        })
        print(f"  Recall@4={s['avg_recall_at_k']:.3f}  MRR={s['avg_mrr']:.3f}  Latency={s['avg_latency_ms']:.1f}ms")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        results.append({**b, "recall": None, "mrr": None, "latency_ms": None, "error": str(exc)})

# Print markdown table
print("\n\n## Benchmark Results\n")
print(f"Dataset: `{DATASET}` · {MAX_QUERIES} queries sampled · generation: `simple`\n")
print("| Chunking | Embedding Model | Recall@4 | MRR | Latency (ms) |")
print("|---|---|---|---|---|")
for r in results:
    recall = f"{r['recall']:.3f}" if r.get("recall") is not None else "ERROR"
    mrr    = f"{r['mrr']:.3f}"    if r.get("mrr")    is not None else "ERROR"
    lat    = f"{r['latency_ms']:.1f}" if r.get("latency_ms") is not None else "ERROR"
    print(f"| {r['chunking']} | {r['embedding']} | {recall} | {mrr} | {lat} |")

# Save JSON for README generation
out = Path("eval/reports/benchmarks.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
