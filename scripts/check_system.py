"""VS Code friendly system diagnostics for the refactored project."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_PATHS = [
    ROOT / "app.py",
    ROOT / "pyproject.toml",
    ROOT / ".vscode" / "launch.json",
    ROOT / ".vscode" / "tasks.json",
    ROOT / "src" / "multimodel_rag" / "core" / "system.py",
    ROOT / "src" / "multimodel_rag" / "processing" / "pdf_processor.py",
    ROOT / "src" / "multimodel_rag" / "retrieval" / "vector_store.py",
]

IMPORTS = [
    "multimodel_rag",
    "multimodel_rag.core.system",
    "multimodel_rag.core.models",
    "multimodel_rag.processing.pdf_processor",
    "multimodel_rag.processing.specialized_chunker",
    "multimodel_rag.retrieval.smart_retriever",
    "multimodel_rag.retrieval.vector_store",
    "multimodel_rag.formatting.advanced_formatter",
    "multimodel_rag.guardrails.self_rag_validator",
]


def check_python() -> bool:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    return sys.version_info >= (3, 10)


def check_paths() -> bool:
    ok = True
    for path in REQUIRED_PATHS:
        exists = path.exists()
        print(f"{'OK' if exists else 'MISSING'}  {path.relative_to(ROOT)}")
        ok &= exists
    return ok


def check_imports() -> bool:
    ok = True
    for name in IMPORTS:
        try:
            importlib.import_module(name)
            print(f"OK      import {name}")
        except Exception as exc:
            print(f"FAIL    import {name} -> {exc}")
            ok = False
    return ok


if __name__ == "__main__":
    print("=" * 72)
    print("MultiModel RAG diagnostics")
    print("=" * 72)
    a = check_python()
    b = check_paths()
    c = check_imports()
    print("=" * 72)
    if a and b and c:
        print("All checks passed.")
        raise SystemExit(0)
    print("Some checks failed.")
    raise SystemExit(1)
