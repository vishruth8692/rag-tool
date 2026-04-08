from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.index.build import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs and build vector index")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    info = build_index(config)
    print(f"Indexed {info['num_docs']} docs into {info['num_chunks']} chunks.")
    print(f"Chunking strategy: {info['chunking_strategy']}")
    print(f"Embedding provider: {info['embedding_provider']}")
    print(f"Index saved to: {info['index_dir']}")


if __name__ == "__main__":
    main()
