# SVI engineering changes

## Confirmed corrections

1. ParamStore isolation deep-copied tensors at every public boundary. A baseline ten-step versus two five-step comparison differs by maximum parameter delta 0.060684; an external Pyro parameter also loses identity after engine use. `_engine_param_store` now swaps private state dictionaries while retaining tensor references under the existing process lock, restores the original external objects and supports nested calls within one engine. Recursive cross-engine reentry is rejected before it can inherit a foreign store. This preserves Adam keys. Regression: exact split/uninterrupted trajectory, external identity, existing concurrent-engine isolation. This defect does not explain E07's uninterrupted trajectory.
2. `posterior_summary` forced covariance samples to float32 and returned a scalar for a one-dimensional site. Baseline float64 values separated by 1e-10 yield zero covariance. It now retains float64, promotes half precision for covariance, returns D×D matrices and accounts for actual tensor element sizes. Nonempty floating samples must be finite. Empty-valued auxiliary sites are skipped. Regression: precision/matrix shape/nonfinite tests. Existing output limits remain in place.

No incorrect SVI objective or single-fit covariance update was demonstrated. No Pyro internals, guide algorithms, likelihood reduction or prior conventions were changed.

## Explicit controls and diagnostics

`core/engine.py`: adds freeze/joint policy (joint default), continuation/reset/restart distinctions, cumulative training history and batch step counts, before/after parameter changes, nonfinite loss/parameter detection and checkpoint policy metadata. Checkpoints remain tensor-only with hashes and declarative constraints. Optimizer continuation is in-process only and a request immediately after checkpoint load fails explicitly.

`core/svi_diagnostics.py`: bounded JSON guide moment inspection with diagonal, site-diagonal, factor-plus-diagonal and Cholesky representations, dense covariance and eigenvalues for small problems, unconstrained site-coordinate metadata, trainable inventory and optimization status. Diagnostics restore RNG state. Custom guides without introspection have an explicit unavailable result. No CLI/MCP execution surface was expanded.

`validation/svi/reference.py`: independent Cholesky Gaussian reference, metrics and direct Pyro model. `scripts/svi_gaussian_benchmark.py`, `svi_sensitivity.py` and `svi_baseline_regressions.py`: source-selected reproducible controls. `tests/test_svi_integrity.py`: focused lifecycle, statistical representation, analytical, objective/gradient parity and numerical regressions. `examples/svi_training_controls.py`: executable freeze/joint/all-selected examples.

## Compatibility and limits

Public positional calls remain accepted. Defaults retain joint deterministic training and fresh optimizer behaviour across fit calls. History now accumulates instead of silently discarding earlier calls. Explicit optimizer resets are observable. Historical research, package version, EXPCON, custom-guide trust checks and MCP restrictions are unchanged. No package was released or pushed.

See POSTERIOR_VALIDATION.md for actual checks and failures. The initial new tests fail on the original API (10 failures, one algebra test passed); baseline_regressions.json separately demonstrates the behavioural failures without depending on new API names. All numerical changes are assessed against equivalent direct Pyro rather than treating an API test as scientific validation.
