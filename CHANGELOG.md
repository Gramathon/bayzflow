# Changelog

## Unreleased

- Document installation from GitHub Releases and test the complete README example.
- Reject non-null `create_plates` in built-in guide specifications and normalize integer `init_scale` values for Pyro.
- Limit the built-in full rank guide to 512 latent coordinates and preserve existing Pyro checkpoint names.
- Document registering the packaged MCP server with Pi.

## 0.2.0

- Added serializable guide specifications and built-in Pyro guide selection.
- Added trusted custom guide registration, local configuration validation and implementation identity.
- Added guide propagation to the Python engine and YAML experiment runner.
- Added private per-engine Pyro parameter-store snapshots guarded by a process-wide lock.
- Added tensor-only safe posterior parameter loading, declarative constraint metadata and guide/checkpoint identity checks.
- Added `bayzflow` diagnostics, guide and packaged-skill commands, plus read-only stdio MCP discovery.
- Raised the Python floor from the inaccurate 3.8 claim to 3.10.
- Corrected version and repository metadata.
- Added bounded Linux CPU jobs through MCP with durable job IDs, cancellation, idempotency and paged tensor artifacts.
- Added native guide covariance factors and an analytical regression fixture for built-in and custom guides.
- Required PyTorch 2.6 or newer for checkpoint loading and documented the verified EXPCON lifecycle.

### Compatibility notes

`setup_svi(guide_kind="low_rank")` and `setup_svi(guide_kind="mean_field")` remain aliases. The experiment runner trains the configured epoch count without validation-based early stopping. A configured checkpoint `outdir` now contains a complete posterior directory (`meta.json`, model and variational tensors, constraint JSON); the old `best_name` single-file behavior is not used. The old `torch.save` Pyro parameter-store checkpoints are not accepted by the safe checkpoint loader. Rebuild the Bayesian model with the original selected layers and prior configuration, then load a checkpoint written by 0.2.0. Python 3.8 and 3.9 are no longer declared supported.
