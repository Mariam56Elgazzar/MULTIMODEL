"""Architecture smoke test for import stability after the refactor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULES = [
    "multimodel_rag.core.models",
    "multimodel_rag.guardrails.hallucination_guard",
    "multimodel_rag.formatting.response_formatter",
    "multimodel_rag.retrieval.vector_store",
    "multimodel_rag.processing.specialized_chunker",
    "multimodel_rag.core.system",
]


def main() -> int:
    for name in MODULES:
        __import__(name)
        print(f"OK import {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
