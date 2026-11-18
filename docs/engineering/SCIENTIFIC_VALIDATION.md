# Scientific validation

```bash
python -m bayzflow.examples.guide_benchmark --output /tmp/bayzflow-benchmark
```

The benchmark ran on Python 3.12.13, PyTorch 2.15.0+cpu and Pyro 1.9.2. It ran 1,800 SVI steps per guide with seeds 7, 19 and 43 and drew 512 posterior samples per fit after reload. The two-feature Gaussian regression uses a standard normal weight prior and fixed observation variance 0.25. Its exact posterior mean is approximately (0.418, 0.116), with diagonal variances (0.384, 0.485) and cross-covariance −0.403. The regression features' correlation makes the covariance comparison informative.

| Guide | Mean L2 error | Covariance Frobenius error | Predictive epistemic variance RMSE | Initial → final ELBO loss | Fit seconds |
|---|---:|---:|---:|---:|---:|
| AutoNormal | 0.044 | 0.779 | 0.066 | 3.651 → 3.401 | 4.62 |
| AutoDiagonalNormal | 0.081 | 0.776 | 0.058 | 4.356 → 3.386 | 4.25 |
| AutoLowRankMultivariateNormal, rank 1 | 0.099 | 0.181 | 0.016 | 4.024 → 2.345 | 6.11 |
| AutoMultivariateNormal | 0.107 | 0.053 | 0.010 | 4.127 → 2.310 | 5.35 |
| Custom bivariate Cholesky guide | 0.098 | 0.075 | 0.016 | 2.948 → 2.322 | 5.03 |

Numbers are averages over seeds. ELBO losses average the first and last 100 steps. The maximum mean error over all 15 fits was 0.1584. The diagonal guides approximated the mean on this dataset while missing weight correlation. The custom guide defines its own Pyro sample site and trainable Cholesky factor, and went through registration, fitting, checkpoint reload, prediction, sampling and artifact persistence. The complete per-fit results include parameter and artifact identities in [gaussian-benchmark.json](evidence/gaussian-benchmark.json).

This exercise tests one tractable likelihood and architecture. Observation noise is known here; the reported predictive metric measures the variance of the conditional mean over weight draws. The experiment cannot establish general calibration, adequate convergence on larger problems, or GPU behavior. Native Python covariance representations are optional and expressed in packed unconstrained coordinates. The benchmark compares empirical covariance in the two weight coordinates.
