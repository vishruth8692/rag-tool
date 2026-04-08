from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ab_testing.runner import run_ab


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B evaluation")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    report = run_ab(config_a=args.config_a, config_b=args.config_b, dataset_path=args.dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
