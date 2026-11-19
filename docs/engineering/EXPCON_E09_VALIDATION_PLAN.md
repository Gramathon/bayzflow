# Proposed E09 independent validation — not executed

This is a proposed successor specification, not a frozen contract or EXPCON verification. E07 remains NOT_SUPPORTED and E08 adjudications are unchanged. EXPCON must independently inspect, validate and freeze its eventual contract before execution; this engineering task does not modify EXPCON.

Hypotheses: (H1) matched Bayzflow and direct Pyro optimize equivalent objectives and produce matching posterior moments; (H2) covariance recovery depends on guide restrictions and optimization geometry; (H3) freeze/joint and optimizer continuation have the documented integrity properties. Structural covariance causality, dataset attribution and broad uncertainty calibration are outside scope.

Baselines: original installed v0.2.0 engine SHA256 `3f11752ad464df98a83712cc4b5e30063e87225d5237bd82b2940d424979cb28`; original checkout commit `3a559b5343839c0dbb91cbdd1d05a4cb40925b25`; independently frozen modified source tree; direct Pyro 1.9.2. Use one pinned Python/Torch/NumPy environment for source comparisons, and record a separate replay under the original Torch environment. Source paths and hashes must be validated before imports.

Dataset identity: `validation/svi/reference.py::dataset`, NumPy SeedSequence([seed,811]), seeds 11/23/37, 48 observations, features z and .92*z + Normal(0,.12), augmented intercept, truth [.8,-.45,.2], noise SD .25. Prior centers derive from float32 nn.Linear(2,1) initialized under seed+8900; prior SD .5. Hash every input array including original float32-rounded design/labels and prior mean. Distinguish this rounding-controlled target from E08's original float64 reference design. Retain actual arrays so library RNG changes cannot masquerade as matching datasets.

Exact reference: use Cholesky posterior precision solves; independently verify residual equations and positive definiteness. Full-data summed Gaussian NLL. Compare conditioned log objectives and gradients at fixed weights in float64 to tolerance 1e-10. Record exact moments, trace/logdet, eigensystem and coordinate transform matrices.

Guides: AutoNormal, AutoDiagonalNormal, rank-2 AutoLowRankMultivariateNormal and AutoMultivariateNormal, init_scale .1. Match initial guide tensors, RNG, precision, Adam LR .02 and full-batch order. Continuous checkpoints 0/100/400/1000/2000, without observer RNG perturbations. One-factor controls: LR .005/.05, init_scale .5, standardized/whitened features with transformed prior and native-coordinate recovery, prior SD .25/1 and float64. Prior changes require recomputing the exact reference.

Acceptance criteria declared before execution:

- Source parity: initial and checkpoint means/covariances match direct Pyro within 1e-5 absolute and relative float32 tolerances, 1e-10 float64; record every failed arm.
- Integrity: frozen parameters bitwise unchanged; joint point parameters and intended guide tensors change; separate engine/global store identities remain isolated. Split continuation equals uninterrupted training bitwise on the fixed CPU stack. Explicit reset yields Adam step 1 after one update.
- Sampling: use 8192 joint native draws, at least three sampling seeds; normalized mean discrepancy <=.06 and covariance relative discrepancy <=.08 against guide moments in at least 2/3 sampling seeds. Report all draws/failed gates. These test internal sampling, not target accuracy.
- Exact posterior adequacy: normalized mean error <=.10, relative covariance error <=.10 and predictive epistemic variance error <=.10 in at least 2/3 data seeds. Report Gaussian KL separately. Parity may pass while adequacy fails.
- Coordinates: invertible reparameterizations preserve exact native mean/covariance and predictions to 1e-10 float64. Empirical moments must agree after transforming joint draws.
- Safety: existing sample/dimension/byte caps, safe checkpoint checksums and trusted-guide restrictions pass. Across-process optimizer continuation is unsupported and must fail explicitly.

Proposed compute cap: 1800 seconds CPU, one Torch thread, sequential jobs; include startup, failures and replays in a durable ledger. Stop at cap with partial execution status, never promote incomplete results. Longer schedules require a separately justified frozen successor, not adaptive running until a gate passes.

Required evidence: contract/source/environment identities; raw arrays and hashes; initial/final variational tensors; native and empirical covariance; all checkpoint metrics and loss curves; actual runtimes; training inventories/deltas; reset/continuation traces; test logs; failure ledger; checksums and EXPCON run/evidence/adjudication identities. Held-out calibration claims require a separately frozen held-out design; current fitting-row coverage is descriptive only.
