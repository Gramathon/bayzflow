# Posterior fidelity investigation

## Independent reference and parity

`validation/svi/reference.py` contains a standalone Gaussian posterior precision-Cholesky solution, KL and covariance metrics and a direct Pyro model with no Bayzflow imports. It avoids explicit covariance inversion. `scripts/svi_gaussian_benchmark.py` matches data, Normal prior centers/scales, full-data summed Gaussian likelihood, native weight/bias sites, guide initialization, Adam learning rate, random seeds and precision. Initial guide evaluation and all checkpoints occur inside one continuous optimizer trajectory. No posterior sample noise enters guide-moment accuracy metrics.

Three E07-style datasets use seeds 11/23/37, 48 strongly correlated observations, noise SD .25, initialized prior center and prior SD .5. Initial scale is .1, learning rate .02. Both paths explicitly initialize the guide before fitting, consuming identical draws. Baseline source, installed v0.2.0 and modified source are compared under the same Torch 2.15.0 interpreter to isolate source effects. This is not a byte-identical E08 replay: initial observation of the guide shifts the RNG schedule, and analytical references use the actual rounded float32 likelihood arrays rather than the original double generator arrays. Historical E08 numbers/verdicts remain unchanged.

For every completed comparison, the maximum checkpoint discrepancy in mean error, covariance relative error, Gaussian KL and predictive variance error is recorded in summary.json and checked against absolute 1e-5. Default full-rank source comparisons match exactly, including their poor covariance recovery. There is no demonstrated extra Bayzflow approximation error in these uninterrupted controls.

## Numerical controls

All rows below are 2000 steps; covariance errors are relative Frobenius norms. Means and covariance are measured separately. Runtime, full ELBO curves, covariance matrices, eigenvectors, correlation matrices and trace/logdet are retained in the result JSON, not collapsed into scalar uncertainty summaries. Sampling consistency is independently tested with 3000 joint native draws per guide family. Predictive coverage and log likelihood here are fitting-row Gaussian predictive diagnostics and do not establish held-out calibration.

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

## Mechanism and unresolved explanations

The original full-rank 400-step covariance errors are .9241/.8785/.9102; at 2000 they are .6258/.5200/.5166. Mean errors at 2000 are .03750/.03803/.03745. Full-rank leading-direction squared agreement is .9984/.9899/.9996. Its largest eigenvalues are .0428/.0567/.0484, below the analytical broad direction. Thus most remaining covariance error concerns insufficient variance recovery, despite substantially correct leading orientation. The arbitrary eigendirection of the initial isotropic covariance has no scientific interpretation.

Diagonal families cannot represent the strongly correlated target; this is a family restriction, separate from optimization. Full-rank Gaussians can represent this exact Gaussian target, so their finite-budget failure cannot be ascribed to a structural Gaussian-family restriction. Low-rank rank 2 has enough flexibility to approach this three-dimensional Gaussian, but finite training need not find that solution.

Learning rate, initialization and conditioning controls change one factor against the matched baseline. Standardization here scales columns by their sample standard deviations without centering. Whitening uses the two-feature SVD; it retains the intercept, transforms the prior exactly and maps guide mean/covariance back to original parameter coordinates. Because the starting prior is isotropic, the rotated/rescaled prior remains diagonal; it is explicitly installed in the benchmark's prior buffers. This is an experimental reparameterization, not a new inference default. Prior SD .25/1 arms each recompute their own exact target. Float64 recomputes both model and likelihood in float64 from the generated inputs.

Optimization geometry and stochastic finite-budget convergence are supported contributors. The experiments do not uniquely apportion their interactions or establish a universal optimizer bias. No amount of API hardening is claimed to improve an unchanged single-fit trajectory. No longer schedule was chosen adaptively until a pass occurred.

Ruled out within the matched Gaussian controls: a Bayzflow-specific objective discrepancy, unintended deterministic backbone, posterior extraction as the sole cause (errors use guide-native moments), and Monte Carlo noise as the sole cause. E08 independently verified observed Normal versus factor likelihood/gradient parity and unchanged frozen parameters. Dropping a fixed Normalizer is not an explanation. Repeated-fit identity loss is real but cannot explain the uninterrupted E07/E08 trajectory. Nonlinear exact-posterior fidelity, broad calibration and dataset-causal interpretations remain unverified.

## Reproduction

See validation/svi/README.md for commands, exact configurations, limitations and acceptance criteria. Failed engineering attempts included unavailable pytest, absent optional SciPy during broad collection, an AutoNormal attribute-access error in the new utility, and benchmark model reuse after in-place wrapping. They were corrected before retaining successful numerical results. No failed attempt is represented as verified scientific evidence. No EXPCON experiment was executed.

Whitening improves covariance recovery from .517–.626 to .133–.232 after mapping to native coordinates, but median absolute mean error rises to .204: the joint accuracy gate still fails. Increasing initial scale to .5 yields .793–.839 covariance error, worse than .1. Column standard-deviation scaling gives little benefit (.530–.622). Float64 alone leaves errors .489–.633. Prior SD .25 improves recovery to .281–.375 against its different exact target; prior SD 1 is reported in the table rather than treated as an interchangeable posterior. These are bounded configuration sensitivities, not changes to statistical defaults.

An independent algebraic check separates the diagonal-family limitation from optimizer failure: the exact reverse-KL-optimal diagonal Gaussian has mean mu_p and variances 1 / diag(Sigma_p^{-1}). Its relative covariance errors here are .98988/.98790/.98529 (meanfield_limit.json), close to the trained diagonal-family errors. Even perfect reverse-KL optimization within that family cannot reproduce this target covariance. This argument does not apply to the full-rank family.
