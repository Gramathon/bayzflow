# Implementation report

**Target:** 0.2.0 candidate
**Baseline:** `30d18ff5792306a64aa53dd5e483bff292152a24`
**Branch:** `feature/bayzflow-v0.2-mcp-guides`

## Implemented in this candidate

- `GuideSpec` supports JSON serialization and stable SHA256 identity. `GuideRegistry` validates constructor kwargs, exposes the four installed Pyro autoguides and registers trusted factories with optional typed schemas, versions, capabilities, validators and compatibility hooks.
- `BayzFlow` accepts a guide and a local registry. `setup_svi`, `auto`, `train_posterior`, and `load_posterior` use that configuration. Legacy guide names remain accepted. `from_model_and_data` performs selected-layer prior calibration and model wrapping without training.
- YAML `guide.kind` and `guide.kwargs` now reach the experiment engine. The runner uses the engine's actual setup/fit signatures and reports guide identity in returned posterior metadata.
- Engine operations take private snapshots of the global Pyro parameter store and serialize access with a process-wide reentrant lock. This protects Bayzflow instances from interleaved in-process operations; it is not process isolation.
- Bayesian layer prior location/scale tensors are module buffers so model-state checkpoint save/reload preserves them. Posterior variational tensors are stored separately from JSON constraint metadata. Loading rejects unrecognized constraints and uses `torch.load(weights_only=True)`. Bounded posterior-site sampling, site summaries and tensor-artifact save/load carry site names, shapes, dtypes, seed, guide/model/checkpoint identity and checksums.
- A `bayzflow` CLI supports version, doctor, built-in guide list/inspect/validate, and packaged skill status/install. The official MCP SDK v2 stdio server currently offers read-only doctor, registered model inspection/candidate discovery, and guide list/schema/validation.
- Added a packaged skill with seven workflow references and documentation about model scope, security and EXPCON handoff.
- Added focused engine, experiment-runner and interface tests, CI workflow, packaging metadata cleanup and release/security/contribution documentation.

## Validation executed

- Python 3.10.12; PyTorch 2.4.1+cu121 with `torch.cuda.is_available() == False`; Pyro 1.9.1; pytest 9.0.2; MCP SDK 2.3.0 (installed under `/tmp/bayzflow-mcp-libs` for isolated protocol testing).
- `PYTHONPATH=/tmp/bayzflow-mcp-libs:$PYTHONPATH python3 -m pytest -q tests/test_guides_engine.py tests/test_interfaces.py tests/test_experiment_runner.py`: 14 passed.
- `PYTHONPATH=/tmp/bayzflow-mcp-libs:$PYTHONPATH python3 -m pytest -q --ignore=tests/test_phase8.py`: 84 passed.
- Ruff on changed engine, experiment, guide, CLI, MCP and focused test files: clean. `mypy bayzflow/guides bayzflow/mcp`: 6 source files, no issues.
- The unfiltered `python3 -m pytest -q` suite does not collect because the pre-existing untracked `tests/test_phase8.py` imports `posterior.weight`, but the checkout resolves `posterior` as a module rather than a package. The phase 8 work was not changed.
- Both distributions built with `python3 -m build`; wheel installed outside the checkout in `/tmp/bayzflow-clean-venv`. CLI, skill access, example inference and an MCP stdio subprocess were exercised from that installed wheel.

## Not implemented / release blockers

Operational MCP fit/sample/measurement tools, durable jobs, resource admission and artifact service remain absent. There is no full EXPCON frozen-contract example executed against the current EXPCON checkout. No analytical Bayesian benchmark, Python 3.11/3.12 run, GPU validation, vulnerability/secret scan, SBOM/provenance, or publication was completed. The environment contains pre-existing unrelated research files; they are excluded from the implementation commit.
