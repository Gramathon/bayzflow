# MCP validation

```bash
python -m pytest -q tests/test_interfaces.py tests/test_mcp_operations.py
```

The complete installed-wheel test run passed 22 tests on Python 3.12.13, PyTorch 2.15.0+cpu, Pyro 1.9.2 and MCP SDK 2.3.0. An official SDK stdio client launched the installed server from a temporary working directory. It planned and fitted the registered regression workload, replayed a fit idempotently, polled status, inspected posterior metadata, and ran posterior sampling and both parameter and predictive measurements. The client reconstructed each artifact from its bounded response and checked SHA256. An in-memory test separately covers discovery and guide validation.

Denial tests cover changed plan IDs, unregistered layer paths, an over-budget model and sample request, concurrent work beyond the limit, unrecognized artifact IDs, symlinked workspaces and checkpoint files, cancellation, and expiration. The job database is reopened to verify persisted state. The service uses one separate Linux process per job, an operator-owned workspace, bounded resource admission, and checked artifact reads by ID. HTTP and GPU workers are outside the validated scope.
