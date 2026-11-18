# MCP validation record

The implementation uses the official `mcp` Python SDK v2 API. It was checked against version **2.3.0**. The source checkout did not have the SDK installed, so the package was installed into the isolated temporary target `/tmp/bayzflow-mcp-libs`; this kept that dependency out of the base user environment.

Command:

```bash
PYTHONPATH=/tmp/bayzflow-mcp-libs:$PYTHONPATH python3 -m pytest -q tests/test_guides_engine.py tests/test_interfaces.py tests/test_experiment_runner.py
```

Result: **14 passed** in the focused engine, experiment and interface suite. The protocol test creates an in-memory official SDK client, lists tools and calls `guide_validate`. The service test covers registered-model inspection/candidate discovery and rejection of an unknown model identifier. A separate installed-wheel subprocess check from `/tmp` used `mcp.Client(StdioServerParameters(...))`, discovered all seven tools and successfully called `guide_validate` over stdio.

Implemented MCP tools: `bayzflow_doctor`, `model_list`, `model_inspect`, `layer_candidates`, `guide_list`, `guide_schema`, and `guide_validate`. Discovery is in-process and read-only. No training is triggered.

Not implemented or validated: `measurement_list`, `posterior_inspect`, MCP artifact tools, controlled fit/sample/measurement operations, job lifecycle, idempotency/cancellation, resource budgets, admission control, workspace/symlink confinement, HTTP transport/authentication, and worker-process isolation. No MCP operational tool is exposed, so these are unclosed release requirements, not advertised capabilities.
