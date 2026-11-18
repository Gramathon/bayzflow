# Baseline audit

**Audited:** 2026-11-18
**Repository:** `/data/dev/repos/VScode/bayzflow`
**Baseline commit:** `30d18ff5792306a64aa53dd5e483bff292152a24`
**Implementation branch:** `feature/bayzflow-v0.2-mcp-guides`

## Working-tree preservation

At baseline the checkout was already on `research/structural-epistemics` with local work in `bf_ckpt/meta.json` and untracked observability, research phase 7/8, and test files. Those paths were preserved while creating the requested branch. Verify with `git status --short --branch`.

## Reproduction commands and findings

Run the following from the repository root:

```bash
git rev-parse HEAD
git status --short --branch
rg -n 'version|requires-python|yourusername' pyproject.toml bayzflow/__init__.py
rg -n 'from_model_and_data|def setup_svi|def fit|def exp|guide_kind' bayzflow/core
rg -n 'guide:' bayzflow/bayzflow.yaml
```

- `pyproject.toml` declared 0.1.1 while `bayzflow/__init__.py` declared 0.1.0. Project URLs pointed at `yourusername`.
- Package metadata advertised Python >=3.8 even though the source uses PEP 604 unions. The implementation baseline is Python 3.10.12; this campaign sets the declared floor to 3.10.
- `BayzFlow.setup_svi()` accepted only `low_rank` and `mean_field`; it directly constructed `AutoLowRankMultivariateNormal(rank=20)` or `AutoDiagonalNormal`. YAML already contained `guide.kind`/`guide.kwargs`, but no propagation reached the engine.
- `BayzFlow.from_model_and_data()` did not exist. `BayzFlow.exp()` called it, then invoked `setup_svi(loss_fn=...)` and `fit()` with unsupported validation/checkpoint arguments. This path could not complete against the engine signatures.
- `train_posterior()` always used the default guide and checkpoint metadata omitted guide identity/configuration. `load_posterior()` rebuilt the default guide.
- Pyro uses a process-global parameter store. The original setup cleared it and the original persistence path used Pyro's unrestricted `torch.load` load routine.
- The repository had no guide registry, packaged MCP server, CLI, agent skill, or focused guide tests at the baseline commit. Existing tests were research analyses and not a tiny library inference acceptance test.

## Environment at audit

Python 3.10.12, PyTorch 2.4.1+cu121, Pyro 1.9.1, pytest 9.0.2, PyYAML 6.0.1. The installed Torch build reports CUDA support; this campaign's tiny tests run on CPU. No `mcp` SDK or `build` frontend was installed at audit time.

## EXPCON reference

Inspected the live `Gramathon/expcon` repository and README. The project treats experiment definition/freeze, controlled execution, evidence verification and scientific adjudication as separate phases; it explicitly describes local arbitrary Python workloads as outside a security sandbox. Bayzflow should therefore emit configuration and artifacts for EXPCON to collect, not reimplement its lifecycle. The installed EXPCON `experiment-contract` skill commands shown by its README are `expcon skill status`, `expcon skill path`, and `expcon skill install`.

## Current audit status

The engine and guide-registry work has begun on the feature branch. See `IMPLEMENTATION_REPORT.md` and `RELEASE_READINESS.md` for verified changes and remaining gates. This audit is not evidence that a release is ready.
