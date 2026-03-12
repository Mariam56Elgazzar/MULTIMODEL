# MultiModel RAG — Refactored for VS Code

This version reorganizes the project into a maintainable `src/` layout while preserving the old top-level file names as compatibility shims.

## Preferred layout

```text
src/multimodel_rag/
  app/
  core/
  formatting/
  guardrails/
  infra/
  integrations/
  memory/
  processing/
  prompts/
  retrieval/
  utils/
```

## Run in VS Code

1. Create a virtual environment with the `Create venv` task.
2. Install dependencies with the `Install deps` task.
3. Launch the app with the `Run app` task or the `Streamlit: app.py` debugger.

## Recommended entrypoints

- `app.py` — main Streamlit entrypoint
- `scripts/check_system.py` — architecture diagnostics
- `scripts/smoke_test_chat.py` — import smoke test

## Backward compatibility

Legacy top-level modules such as `enhanced_rag_system.py` and `vector_store.py` are retained as thin wrappers so existing imports do not break immediately.
