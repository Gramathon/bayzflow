---
name: bayesian-inference
description: Use Bayzflow for selective Bayesian inference and bounded posterior uncertainty analysis in supported PyTorch models.
---

# Bayzflow Bayesian inference

Bayzflow replaces selected supported PyTorch layers with Pyro latent-weight layers and fits a variational posterior. It is useful when an experiment needs parameter posterior samples or predictive variation. It does not guarantee calibrated uncertainty or support every architecture.

## Discover the installed capabilities

Run `bayzflow doctor` and `bayzflow guide list`. Inspect a guide with `bayzflow guide inspect auto_lowrank`. Validate a built-in guide YAML file with `bayzflow guide validate path/to/guide.yaml`.

Use the Python API with a trusted model and data loader:

```python
from bayzflow import BayzFlow

bf = BayzFlow.from_model_and_data(
    model=model,
    calib_loader=calibration_batches,
    calib_forward_fn=lambda m, batch: m(batch["x"]),
    bayes_patterns=["head"],
    guide="auto_lowrank",
    guide_kwargs={"rank": 8},
)
bf.train_posterior(
    train_loader=train_batches,
    loss_from_logits_fn=lambda logits, batch: loss(logits, batch["y"]),
    predict_fn=lambda m, batch: m(batch["x"]),
    num_epochs=20,
    save_dir="artifacts/posterior",
)
samples = bf.predict(batch, predict_fn=lambda m, b: m(b["x"]), num_samples=32,
                     return_samples=True)
```

Selected layers must be confirmed with `bf.list_candidates()` and the measured input shapes. Start with a small head or projection layer; full-rank guides scale poorly with latent dimension. Custom guide factories must be implemented and registered in trusted Python code. An agent may select a registered `custom:<name>` but must never supply Python source, an import path, or a serialized callable.

## Preserve evidence and interpret results

Keep the posterior directory, guide specification, selected module names, random seeds, data identity, training configuration, package versions, and checksums with the experiment. `predict` returns sample tensors; `save_posterior_samples` writes bounded tensor artifacts with a metadata sidecar, and `BayzFlow.load_posterior_sample_artifact` verifies and reloads them. Report bounded summaries for large results. Predictive standard deviation combines uncertainty sources and must not be reported as a calibrated epistemic estimate without a separate validation design. Layer activation dispersion and attention entropy are distinct measurements; attention entropy is not epistemic uncertainty.

For EXPCON, declare the Bayzflow invocation, exact source/dependency identity, inputs, seeds, guide and layer configuration, output metrics, posterior tensor artifacts, and their checksums in the frozen experiment. Let EXPCON freeze, execute, collect and verify evidence. A successful run is not itself a scientific verdict.

## Security limits

Use local registered models and trusted checkpoints. Never accept client-provided filesystem paths or executable Python. Posterior loading uses tensor-only checkpoint loading and a bounded declarative constraint file. Pyro has a process-global parameter store; Bayzflow serializes its own engine operations and snapshots per-instance state. Do not run independent Bayzflow work concurrently in threads or claim cross-process isolation without an external worker boundary.

See the reference guides for [guide selection](references/guide-selection.md), [custom guides](references/custom-guides.md), [posterior sampling](references/posterior-sampling.md), [model scope](references/model-selection.md), [uncertainty measurements](references/epistemic-measurement.md), [security](references/security.md), and [EXPCON handoff](references/expcon-integration.md).
