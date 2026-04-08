from src.eval.metrics import mrr, recall_at_k


def test_recall_at_k_hits() -> None:
    got = recall_at_k(["a", "b", "c"], ["b"], k=2)
    assert got == 1.0


def test_mrr_first_hit() -> None:
    got = mrr(["x", "y", "z"], ["y"])
    assert got == 0.5
