from pathlib import Path

from src.common.config import load_config


def test_load_config_with_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text("a:\n  b: 1\n  c: 2\n", encoding="utf-8")
    child.write_text("extends: base.yaml\na:\n  c: 4\n", encoding="utf-8")

    cfg = load_config(str(child))
    assert cfg["a"]["b"] == 1
    assert cfg["a"]["c"] == 4
