# Scientific validation record

## Executed evidence

Environment: Python 3.10.12, PyTorch 2.4.1+cu121, Pyro 1.9.1, CPU execution. Command:

```bash
PYTHONPATH=/tmp/bayzflow-mcp-libs:$PYTHONPATH python3 -m pytest -q tests/test_guides_engine.py tests/test_interfaces.py tests/test_experiment_runner.py
```

Latest recorded result: **14 passed**. The focused tests train a small PyTorch linear model with SVI and exercise `AutoNormal`, `AutoDiagonalNormal`, `AutoLowRankMultivariateNormal`, `AutoMultivariateNormal`, and one locally registered factory that returns an actual Pyro `AutoDiagonalNormal`. They check finite predictive results, bounded posterior-site sampling and summaries, tensor-artifact metadata/checksum round trips, guide metadata, artifact tamper rejection, and checkpoint parameter/model-buffer round trips. A pre-existing `BayesianLSTM` PyroModule and the YAML experiment runner are also trained. The broader suite excluding the untracked phase 8 import failure passed **84 tests**.

The custom guide test uses a two-dimensional latent model and two epochs. Built-in smoke cases use one epoch. The test does not measure calibration, robust convergence, covariance recovery accuracy, uncertainty decomposition or computational scaling. The shared synthetic data are fixed by seed 19; no analytical posterior comparison is included. The store-isolation test verifies two trained engine instances can predict after one another, but no threaded stress test was run.

## Limits

This evidence validates basic construction, a short SVI path, prediction and persistence for a tiny CPU regression fixture only. It does not establish equivalent performance between guides, calibration, model generality, GPU operation, large latent dimensions, or correct uncertainty for any application. The default factor loss path is model- and callback-dependent; observation likelihoods should be expressed with Pyro sample sites when a probabilistic likelihood is needed.

An analytical reference benchmark, multi-seed convergence analysis and measured cost/covariance comparison across guides remain mandatory scientific validation work before a production claim.
