from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    report = run_eval(config_path=args.config, dataset_path=args.dataset)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {report['report_path']}")


if __name__ == "__main__":
    main()
