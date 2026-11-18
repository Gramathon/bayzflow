# Release readiness: 0.2.0

**Status: NOT READY FOR PRODUCTION OR PUBLICATION**
**Candidate artifacts:** local wheel and sdist; not a GitHub Release
**Implementation branch:** `feature/bayzflow-v0.2-mcp-guides`

## Verified

- Python 3.10 CPU scientific smoke: the four built-in guides and one true trusted custom guide train and predict on tiny PyTorch/Pyro models. A pre-existing BayesianLSTM PyroModule trains without being wrapped twice.
- Guide specs, trusted registry, YAML propagation, old guide selector aliases, per-engine Pyro parameter snapshots, prior buffers, posterior-site samples/summaries and safe checkpoint round trips have focused tests.
- A checkpoint tamper test detects modified variational tensor bytes. The loader uses weights-only tensor loading, checks artifact hashes and checkpoint identity, and restores guide identity/configuration.
- Focused engine/interface/experiment-runner tests: **14 passed** with MCP SDK 2.3.0 available. The broader suite excluding one pre-existing untracked test: **84 passed**. Ruff and Mypy checks on changed modules pass.
- Wheel and sdist build; installed-wheel CLI, skill packaging/install, and official MCP stdio subprocess smoke checks pass outside the source checkout.
- Current package metadata and Python floor agree on version 0.2.0 and Python >=3.10.

## Mandatory blockers

1. The complete working-tree test command fails at collection on pre-existing untracked `tests/test_phase8.py` (`posterior.weight`: `posterior` resolves as a non-package module). The related research work was left untouched.
2. MCP only implements read-only discovery and guide validation. Required operational fit/sample/measure tools, durable job IDs/status, resource admission, budget enforcement, cancellation/idempotency, MCP artifact APIs, and workspace/symlink controls are absent.
3. There is no analytical posterior benchmark or multi-seed convergence/calibration study. The current tiny tests are API/persistence smoke evidence only.
4. No full EXPCON frozen-contract and evidence-verification example was executed; the integration is documented but not demonstrated against a specific current schema.
5. Python 3.11/3.12 CI execution has not run; GPU validation is unavailable (`torch.cuda.is_available()` returned false).
6. Dependency vulnerability audit, secret scanning, broader static security review, SBOM, build provenance/attestation, and pinned third-party action review remain incomplete.
7. Local artifacts have not been published or installed from a GitHub Release. No release authorization was provided, and publication remains intentionally undone.

## Release decision

Do not call this candidate production-ready and do not publish it. It is suitable for continued review of the guide registry, CPU inference path, local CLI, discovery-only MCP server and packaged skill. Close every blocker above, rerun validation on the exact candidate artifacts, and obtain explicit release authorization before publication.

Artifact names and local hashes are recorded in `PACKAGING_VALIDATION.md`.
