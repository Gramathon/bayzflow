# Contributing

Use Python 3.10 or newer. Install the development dependencies with:

```bash
python -m pip install -e '.[dev,mcp]'
```

Run the focused inference and interface suite with:

```bash
python -m pytest -q tests/test_guides_engine.py tests/test_interfaces.py
```

Run Ruff on new guide, CLI and MCP code with:

```bash
ruff check bayzflow/guides bayzflow/cli.py bayzflow/mcp tests/test_guides_engine.py tests/test_interfaces.py
```

Changes to inference should include a small real Pyro run, posterior sampling, and persistence coverage. Do not claim calibration or general architecture support from a smoke test. MCP tools must use explicitly registered local objects, bound output sizes, and avoid arbitrary paths, imports, subprocesses or executable serialization. Keep optional example dependencies out of the core dependency set.

The repository may contain active research work in the working tree. Stage only files relevant to your change. Follow the project's concise lowercase present-tense commit style.
