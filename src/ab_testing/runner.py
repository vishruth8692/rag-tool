from __future__ import annotations

import random
from collections.abc import Callable

from src.eval.runner import run_eval


def _bootstrap_p_value(deltas: list[float], n_samples: int = 1000) -> float:
    if not deltas:
        return 1.0
    observed = sum(deltas) / len(deltas)
    centered = [d - observed for d in deltas]
    count = 0
    for _ in range(n_samples):
        sample = [random.choice(centered) for _ in centered]
        mean_sample = sum(sample) / len(sample)
        if abs(mean_sample) >= abs(observed):
            count += 1
    return count / n_samples


def run_ab(
    config_a: str,
    config_b: str,
    dataset_path: str,
    metric_key: str = "faithfulness_proxy",
    max_queries: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    emit("Running control evaluation")
    report_a = run_eval(config_a, dataset_path, max_queries=max_queries, progress_callback=lambda m: emit(f"Control: {m}"))
    emit("Running test evaluation")
    report_b = run_eval(config_b, dataset_path, max_queries=max_queries, progress_callback=lambda m: emit(f"Test: {m}"))

    qa = report_a["queries"]
    qb = report_b["queries"]
    n = min(len(qa), len(qb))
    if n and metric_key not in qa[0]:
        raise ValueError(f"Unsupported A/B metric `{metric_key}`")

    deltas = [qb[i][metric_key] - qa[i][metric_key] for i in range(n)]
    mean_delta = (sum(deltas) / n) if n else 0.0
    p_value = _bootstrap_p_value(deltas)

    winner = "B" if mean_delta > 0 else "A"
    if p_value > 0.05:
        winner = "Inconclusive"

    emit("Computing A/B significance")
    return {
        "config_a": config_a,
        "config_b": config_b,
        "dataset": dataset_path,
        "metric": metric_key,
        "mean_delta_b_minus_a": mean_delta,
        "p_value": p_value,
        "winner": winner,
        "summary_a": report_a["summary"],
        "summary_b": report_b["summary"],
        "report_path_a": report_a.get("report_path"),
        "report_path_b": report_b.get("report_path"),
    }
