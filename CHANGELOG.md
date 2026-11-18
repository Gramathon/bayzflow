# Changelog

## 0.2.0 — unreleased candidate

- Added serializable guide specifications and built-in Pyro guide selection.
- Added trusted custom guide registration, local configuration validation and implementation identity.
- Added guide propagation to the Python engine and YAML experiment runner.
- Added private per-engine Pyro parameter-store snapshots guarded by a process-wide lock.
- Added tensor-only safe posterior parameter loading, declarative constraint metadata and guide/checkpoint identity checks.
- Added `bayzflow` diagnostics, guide and packaged-skill commands, plus read-only stdio MCP discovery.
- Raised the Python floor from the inaccurate 3.8 claim to 3.10.
- Corrected version and repository metadata.

### Compatibility notes

`setup_svi(guide_kind="low_rank")` and `setup_svi(guide_kind="mean_field")` remain aliases. The experiment runner trains the configured epoch count without validation-based early stopping. A configured checkpoint `outdir` now contains a complete posterior directory (`meta.json`, model and variational tensors, constraint JSON); the old `best_name` single-file behavior is not used. The old `torch.save` Pyro parameter-store checkpoints are not accepted by the safe checkpoint loader. Rebuild the Bayesian model with the original selected layers and prior configuration, then load a checkpoint written by 0.2.0. Python 3.8 and 3.9 are no longer declared supported.

This is not yet a release. See `docs/engineering/RELEASE_READINESS.md` for open gates.
