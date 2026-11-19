# Posterior validation

The numerical comparison covers independent analytical Gaussian inference, direct Pyro, original checkout, original installed v0.2.0 source and modified checkout. Same-interpreter source comparisons isolate implementation from dependency differences. Per-run environment/path/SHA256 and elapsed runtime are retained. The original E08 environment remains separately identified in SVI_BASELINE_AUDIT.md.

| Condition | Guide | Mean error median | Relative covariance errors (11 / 23 / 37) | KL median | Predictive variance error median |
|---|---|---:|---|---:|---:|
| baseline | auto_fullrank | 0.0375 | 0.6258 / 0.5200 / 0.5166 | 1.0804 | 0.3048 |
| float64 | auto_fullrank | 0.0373 | 0.6330 / 0.5750 / 0.4892 | 0.8554 | 0.2495 |
| guide_families | auto_diag | 0.0183 | 0.9906 / 0.9895 / 0.9871 | 2.1434 | 0.5658 |
| guide_families | auto_fullrank | 0.0375 | 0.6258 / 0.5200 / 0.5166 | 1.0804 | 0.3048 |
| guide_families | auto_lowrank | 0.0225 | 0.8047 / 0.7547 / 0.7008 | 0.6901 | 0.4053 |
| guide_families | auto_normal | 0.0321 | 0.9861 / 0.9883 / 0.9856 | 2.8072 | 0.6759 |
| init05 | auto_fullrank | 0.0485 | 0.8388 / 0.7932 / 0.7936 | 0.6793 | 0.4909 |
| installed | auto_fullrank | 0.0375 | 0.6258 / 0.5200 / 0.5166 | 1.0804 | 0.3048 |
| lr005 | auto_fullrank | 0.1486 | 0.9087 / 0.8463 / 0.8621 | 0.9089 | 0.3542 |
| lr05 | auto_fullrank | 0.0487 | 0.5807 / 0.5090 / 0.4982 | 0.8390 | 0.1945 |
| modified | auto_fullrank | 0.0375 | 0.6258 / 0.5200 / 0.5166 | 1.0804 | 0.3048 |
| prior025 | auto_fullrank | 0.0378 | 0.3747 / 0.3082 / 0.2811 | 0.9297 | 0.2055 |
| prior1 | auto_fullrank | 0.0380 | 0.7369 / 0.6318 / 0.6197 | 1.1670 | 0.3664 |
| standardized | auto_fullrank | 0.0394 | 0.6224 / 0.5448 / 0.5297 | 1.2098 | 0.3062 |
| whitened | auto_fullrank | 0.2042 | 0.1330 / 0.2246 / 0.2317 | 0.6016 | 0.1543 |

## Verification

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_svi_integrity.py tests/test_guides_engine.py tests/test_interfaces.py tests/test_experiment_runner.py -q
.venv/bin/python scripts/svi_sensitivity.py
.venv/bin/python scripts/svi_summarize.py
git diff --check
```

Focused package tests: **31 passed, 1 skipped** (optional MCP protocol dependency unavailable). All tracked tests plus the new integrity tests: **80 passed, 1 skipped, 1 failed**. The failure is `test_phase5_1.py::test_frozen_phase5_pairs_and_outputs_unchanged`, a frozen parameter-direction identity under this Torch environment. It reproduces against untouched baseline source. Its artifact/hash expectation was not altered. Plain pytest additionally cannot collect pre-existing untracked phase7/8 research tests because their local modules collide; this is recorded, not hidden by changing those tests.

Assertions cover analytical posterior algebra and nonorthogonal coordinate recovery; conditioned objective/gradient equality to 1e-12 float64; bitwise split-fit continuation and ParamStore identities; freeze/joint changes; guide updates; explicit reset/restart; all four guide moment representations; Monte Carlo native-moment consistency; scalar-site covariance shape and float64 precision; nonfinite-state rejection; dimension limits checked before fresh full-rank allocation; recursive cross-engine reentry protection; checkpoint model/guide recovery and checksums; loaded-optimizer limitation and training-policy mismatch; concurrent engine isolation and external store identity; unchanged CLI/MCP restrictions. Sampling mean discrepancy threshold is .08 normalized by sqrt(trace covariance), covariance relative discrepancy .12 for fixed 3000 draws. These tests do not assert exact-target adequacy from samples.

`baseline_regressions.json` independently shows pre-change lifecycle and precision failures. Baseline split-fit parameter discrepancy is .060684; tiny float64 covariance is rounded to zero and represented as a scalar; external parameter identity is lost. New tests verify corrected behaviour.

## Supported claims

Bayzflow matches direct Pyro on these objectives and trajectories. Native guide covariance, sampled native-weight covariance, activation covariance and predictive covariance are distinct estimands. Small guide changes or stable ELBO do not establish posterior correctness. The default original-feature full-rank 2000-step schedule remains inadequate at a .10 relative covariance gate; the engine fixes do not change that result. Parameterization/initialization improvements belong to explicit scientific configuration, not silent posterior changes. Fitting-row predictive diagnostics do not establish coverage on new observations. No nonlinear calibration, causal dataset interpretation or EXPCON verification is claimed.

Release validation remains incomplete because the full test suite is not green, optional MCP protocol testing is skipped, and a pinned multi-version/device compatibility matrix has not been run. Independent E09 validation should precede a production release.

The main-based PR port has a separate full-suite result: **50 passed, 2 skipped**. See [SVI_PR_VALIDATION.md](SVI_PR_VALIDATION.md). This supersedes the original checkout's regression blocker for the PR branch only; independent scientific validation and compatibility checks remain outstanding.
