from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from src.common.config import load_config
from src.common.io import read_jsonl
from src.eval.metrics import faithfulness_bertscore_batch, faithfulness_proxy, mrr, recall_at_k
from src.index.build import build_index
from src.rag.pipeline import RAGPipeline


def run_eval(
    config_path: str,
    dataset_path: str,
    max_queries: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    emit("Loading config")
    config = load_config(config_path)
    index_dir = Path(config["paths"]["index_dir"])
    vectors_file = index_dir / "vectors.json"
    chunks_file = index_dir / "chunks.json"

    try:
        build_index(config, progress_callback=emit)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build index for config `{config_path}`. "
            f"Check embedding provider dependencies and model settings. Root cause: {exc}"
        ) from exc

    if not vectors_file.exists() or not chunks_file.exists():
        raise RuntimeError(
            f"Index build completed but files are missing in `{index_dir}` "
            "(expected `vectors.json` and `chunks.json`)."
        )

    emit("Initializing retrieval pipeline")
    pipeline = RAGPipeline(config_path=config_path)
    emit("Loading evaluation dataset")
    rows = read_jsonl(dataset_path)

    if max_queries and len(rows) > max_queries:
        import random
        rng = random.Random(42)
        rows = rng.sample(rows, max_queries)
        emit(f"Sampled {max_queries} queries (dataset has {len(rows) + max_queries} total)")

    q_metrics: list[dict] = []
    # Collected separately for batch BERTScore computation after the query loop
    _answers_for_bert: list[str] = []
    _contexts_for_bert: list[str] = []

    total = max(1, len(rows))
    for idx, row in enumerate(rows, start=1):
        if idx == 1 or idx == total or idx % 5 == 0:
            emit(f"Evaluating queries ({idx}/{total})")
        query = row["query"]
        expected = row.get("expected_doc_ids", [])

        start = time.perf_counter()
        output = pipeline.query(query)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [src["doc_id"] for src in output["sources"]]
        recall = recall_at_k(retrieved_ids, expected, k=min(4, max(1, len(retrieved_ids))))
        q_mrr = mrr(retrieved_ids, expected)

        source_texts = []
        retrieval_trace = []
        for src in output["sources"]:
            chunk = next((c for c in pipeline.index.chunks if c["chunk_id"] == src["chunk_id"]), None)
            if chunk:
                source_texts.append(chunk["text"])
                retrieval_trace.append(
                    {
                        "doc_id": src.get("doc_id"),
                        "chunk_id": src.get("chunk_id"),
                        "score": src.get("score"),
                        "preview": chunk["text"][:240] + ("..." if len(chunk["text"]) > 240 else ""),
                    }
                )
        faithful = faithfulness_proxy(output["answer"], source_texts)
        _answers_for_bert.append(output["answer"])
        _contexts_for_bert.append(" ".join(source_texts))

        hallucination = output.get("hallucination", {})
        type1 = hallucination.get("type1", {})
        type2 = hallucination.get("type2", {})
        type2_skipped = bool(type2.get("skipped", False))

        q_metrics.append(
            {
                "query": query,
                "recall_at_k": recall,
                "mrr": q_mrr,
                "faithfulness_proxy": faithful,
                "faithfulness_bertscore": 0.0,   # filled in below after batch compute
                "latency_ms": latency_ms,
                "confidence": output["confidence"],
                "hallucination_risk": output["hallucination_risk"],
                # Type 1 — retrieval failure
                "hallucination_type1_flagged": type1.get("flagged", False),
                "hallucination_type1_risk_level": type1.get("risk_level", "unknown"),
                "hallucination_type1_confidence": type1.get("confidence", output["confidence"]),
                # Type 2 — generation unfaithfulness
                "hallucination_type2_skipped": type2_skipped,
                "hallucination_type2_flagged": type2.get("flagged", False),
                # Use None for score when skipped — don't pollute average with 1.0 placeholders
                "hallucination_type2_score": None if type2_skipped else type2.get("score"),
                "hallucination_type2_entailed": type2.get("entailed", 0),
                "hallucination_type2_total_sentences": type2.get("total_sentences", 0),
                "retrieval_trace": retrieval_trace,
            }
        )

    # Batch BERTScore — runs once over all answers, much faster than per-query
    generation_provider = config.get("generation", {}).get("provider", "simple")
    emit("Computing BERTScore faithfulness (batch)")
    bert_scores, bert_fallback, bert_fallback_reason = faithfulness_bertscore_batch(
        _answers_for_bert, _contexts_for_bert
    )
    for i, q in enumerate(q_metrics):
        q["faithfulness_bertscore"] = bert_scores[i]

    n = max(1, len(q_metrics))

    # Type 2 NLI averages — only over queries where NLI actually ran (not skipped)
    type2_ran = [x for x in q_metrics if not x.get("hallucination_type2_skipped", False)]
    type2_scores_valid = [x["hallucination_type2_score"] for x in type2_ran if x["hallucination_type2_score"] is not None]
    type2_nli_skipped_count = sum(1 for x in q_metrics if x.get("hallucination_type2_skipped", False))

    summary = {
        "config": config_path,
        "dataset": dataset_path,
        "num_queries": len(q_metrics),
        "generation_provider": generation_provider,
        "avg_recall_at_k": sum(x["recall_at_k"] for x in q_metrics) / n,
        "avg_mrr": sum(x["mrr"] for x in q_metrics) / n,
        "avg_faithfulness_proxy": sum(x["faithfulness_proxy"] for x in q_metrics) / n,
        "avg_faithfulness_bertscore": sum(x["faithfulness_bertscore"] for x in q_metrics) / n,
        "bertscore_fallback": bert_fallback,
        "bertscore_fallback_reason": bert_fallback_reason,
        "avg_latency_ms": sum(x["latency_ms"] for x in q_metrics) / n,
        "avg_confidence": sum(x["confidence"] for x in q_metrics) / n,
        "hallucination_rate": sum(1 for x in q_metrics if x["hallucination_risk"]) / n,
        # Broken-down hallucination rates
        "type1_hallucination_rate": sum(1 for x in q_metrics if x["hallucination_type1_flagged"]) / n,
        "type2_hallucination_rate": (
            sum(1 for x in type2_ran if x["hallucination_type2_flagged"]) / len(type2_ran)
            if type2_ran else None
        ),
        "avg_type2_nli_score": (
            sum(type2_scores_valid) / len(type2_scores_valid)
            if type2_scores_valid else None
        ),
        "type2_nli_skipped_count": type2_nli_skipped_count,
        "type2_nli_ran_count": len(type2_ran),
    }

    report = {
        "summary": summary,
        "queries": q_metrics,
    }

    emit("Saving evaluation report")
    out_dir = Path(config["paths"]["eval_reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    out_path = out_dir / f"eval_{Path(config_path).stem}_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    report["report_path"] = str(out_path)
    emit("Evaluation complete")
    return report
