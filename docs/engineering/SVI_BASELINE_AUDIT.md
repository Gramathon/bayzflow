# SVI baseline audit

Audit date: 2026-11-19. Starting commit `3a559b5343839c0dbb91cbdd1d05a4cb40925b25`, branch `feature/bayzflow-v0.2-mcp-guides`. Work proceeds on `engineering/svi-posterior-fidelity`. Existing modified `bf_ckpt/meta.json`, untracked observability/research code, phase7/8 tests and uv.lock are preserved. No historical experiment is edited.

## Source and environment identities

The checkout's `.venv` imports this checkout, distribution metadata 0.2.0, Torch 2.15.0+cu132, Pyro 1.9.2. The research `.venv` imports its own site-packages Bayzflow 0.2.0, Torch 2.14.1+cu126, Pyro 1.9.2. Import identity must be selected explicitly: running Python from the checkout can shadow the installed distribution. Benchmarks record actual engine path and SHA256, Python and dependency versions. A tracked-source snapshot at `/tmp/bayzflow-svi-baseline` preserves the pre-change implementation. Installed and checkout comparisons use the checkout interpreter with explicit source selection, isolating source from Torch-version effects; historical E08 used the research interpreter.

The installed engine's recorded E08 hash is `3f11752ad464df98a83712cc4b5e30063e87225d5237bd82b2940d424979cb28`. The baseline checkout engine SHA256 is `76741b219e1f1299619a36e7544c85e39683f705930956cb1c4df5a280e405ee`; current source hashes are recorded in validation JSON. `guides/_fullrank.py` is absent in this checkout; the registry uses Pyro's AutoMultivariateNormal directly. Do not invent a package-local full-rank implementation.

## Architecture and training path

`core/engine.py` discovers supported child modules by type and substring selection. `attach_priors` replaces selected layers in the supplied model in place. Rebuild from a fresh model when starting another inference experiment. Linear priors use native named weight/bias sample sites, independent Normals centered at the supplied model weights; calibrated or explicit layer scales are buffers. Convolution wrappers use PyroSample. Calibration uses activations and is a modelling choice, not evidence of prior correctness.

`guides/specifications.py` defines finite JSON configuration identities. `guides/registry.py` constructs four Pyro builtins or trusted registered factories, validates arguments and records implementation provenance. No arbitrary import path execution is accepted.

`setup_svi` registers remaining trainable deterministic parameters through `pyro.module(update_module_params=False)`, evaluates one prediction, records logits, and applies `pyro.factor('nll', -callback_loss)`. Adam and Trace_ELBO optimize guide and registered point parameters. The caller owns likelihood reduction and minibatch scaling. The flag does not freeze deterministic parameters. Full-batch summed Gaussian NLL matches the intended conjugate posterior; fixed omitted normalizers do not affect its optimum.

`fit` reuses its SVI instance internally; however every public operation restores a deep-copied ParamStore snapshot. This can change tensor identities used as optimizer keys between calls. `setup_svi` constructs a new guide and optimizer; existing stored guide names can still carry old initialization. Both lifecycle questions require regression evidence before correction.

Predictive draws reconstruct native weight tensor sites and logits. `posterior_summary` computes site covariance with a dimension/byte cap; baseline casts covariance inputs to float32 and returns a scalar for a single-dimensional site via torch.cov. These are extraction concerns, separate from SVI accuracy. Layer measurement hooks provide scalar activation summaries and are not parameter covariance or full activation covariance. No native guide covariance API is present in this checkout.

Checkpoint writing uses tensor-only files, declarative constraint JSON, SHA256 identities, guide provenance and software metadata. Loading enforces checksums and weights_only=True. Optimizer and RNG continuation are not persisted. CLI provides doctor, guide schemas and skill assets; MCP uses restricted experiment operations and resource planning. New diagnostics should remain Python-first and preserve those boundaries.

## Evidence reviewed

Read E07 research, Bayzflow and reproducibility reports; E08 research, SVI audit, fidelity, root-cause, training-regime, EXPCON and reproducibility reports; E08 forensic comparison, protocol and uninterrupted-trajectory implementation. Sources remain in `/data/dev/repos/VScode/bf-pi-test`. Their execution/evidence/adjudication distinctions remain authoritative. E07 remains NOT_SUPPORTED. These engineering runs are not EXPCON verification.

E08 already rules out an accidentally trainable backbone and an incorrect full-batch likelihood for E07. It establishes real guide updates, frozen-backbone integrity and optimizer lifecycle sensitivity. Full-rank covariance errors at 2,000 steps remain 0.620/0.510/0.515. This is evidence of inadequate recovery under that schedule, not proof of an inference defect. Optimization, family restrictions, conditioning and initialization remain hypotheses for independent matched controls.

Installed-source differences include additional `posterior_parameters` and `posterior_covariance` methods and RNG preservation around high-level posterior training. Those methods are absent from the baseline checkout; the new checkout diagnostics do not retroactively describe the installed API. Both sources share the deep-copy isolation behaviour. Native Pyro full-rank classes are used in both tested paths.
