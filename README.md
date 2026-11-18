# Bayzflow

Bayzflow is a Bayesian deep learning framework for selective Bayesian inference and uncertainty analysis in PyTorch models. It uses Pyro variational inference to place posterior distributions over selected supported layers.

The project remains research software. Layer replacement, prior calibration and variational inference are model dependent. Predictive variation is not automatically calibrated uncertainty, and broad architecture compatibility has not been established.

## Install

Python 3.10 or newer is required. Install the core package with:

```bash
python -m pip install bayzflow
```

Optional dependencies are separate: `bayzflow[fx]`, `bayzflow[medical]`, `bayzflow[viz]`, `bayzflow[mcp]`, and `bayzflow[dev]`.

## Quickstart

```python
from bayzflow import BayzFlow

bf = BayzFlow.from_model_and_data(
    model=model,
    calib_loader=calibration_loader,
    calib_forward_fn=lambda m, batch: m(batch["x"]),
    bayes_patterns=["head"],
    guide="auto_lowrank",
    guide_kwargs={"rank": 8},
)
bf.train_posterior(
    train_loader=train_loader,
    loss_from_logits_fn=lambda logits, batch: loss(logits, batch["y"]),
    predict_fn=lambda m, batch: m(batch["x"]),
    num_epochs=10,
    save_dir="artifacts/posterior",
)
result = bf.predict(batch, predict_fn=lambda m, b: m(b["x"]), num_samples=32,
                    return_samples=True)
manifest = bf.save_posterior_samples(result["samples"], "artifacts/predictive-samples.pt",
                                     random_seed=17)
```

Candidate layer wrappers currently cover Linear, Conv1d/2d/3d, and ConvTranspose3d modules. Normalization, ConvTranspose1d/2d and arbitrary composite modules are not candidates. Inspect candidates and verify model behavior on a small dataset before scaling an inference run.

## Guides

Built-in guides are `auto_normal`, `auto_diag`, `auto_lowrank`, and `auto_fullrank`. A JSON-serializable `GuideSpec` can also be passed as `guide=GuideSpec(kind="auto_lowrank", kwargs={"rank": 8})`. Legacy `guide_kind="low_rank"` and `guide_kind="mean_field"` spellings remain accepted by `setup_svi`.

Custom Pyro guide factories are registered by trusted Python code through `GuideRegistry.register`. Serialized configuration selects a local registration; it cannot provide Python code or an import path. See [custom guide notes](docs/guides.md).

## CLI and MCP discovery

`bayzflow --version`, `bayzflow doctor`, `bayzflow guide list`, `bayzflow guide inspect auto_lowrank`, and `bayzflow guide validate guide.yaml` provide diagnostics and guide discovery. The packaged skill is available through `bayzflow skill status` and `bayzflow skill install`.

Install `bayzflow[mcp]` to run `bayzflow-mcp --stdio`. The current MCP server exposes read-only local diagnostics, registered-model inspection, layer candidate discovery, and trusted guide listing/schema/validation. It does not expose training, posterior sampling, measurement, remote HTTP, or arbitrary filesystem operations. Models and custom guides must be registered by trusted host Python before server creation.

## EXPCON

EXPCON owns experiment definition, freezing, execution admission, evidence collection, verification and adjudication. Record Bayzflow source/dependency identity, data and model inputs, selected layers, guide/version, seeds, training configuration, output metrics and posterior tensor artifact checksums in the frozen contract. Bayzflow does not depend on EXPCON at runtime. See [integration notes](docs/engineering/EXPCON_INTEGRATION.md).

## Security and scientific limitations

- Checkpoints must come from a trusted source. Tensor loading uses `weights_only=True`; guide constraint data is declarative JSON and supports only known constraints.
- Pyro maintains process-global parameter state. Bayzflow serializes operations within its process and snapshots per-engine parameters. Parallel execution should use separate worker processes; cross-process isolation is the operator's responsibility.
- Prior scales calibrated from activations are heuristic. They do not establish posterior calibration.
- Predictive standard deviation does not by itself separate epistemic and aleatoric uncertainty.
- Attention entropy is not epistemic uncertainty. No causal interpretation of attention is claimed.
- GPU inference has not been validated by this CPU-focused campaign.

## Release status

Version 0.2.0 is an implementation candidate, not a production release. MCP operational jobs, full scientific comparison, supported Python matrix, CI/release automation, and clean wheel installation remain release gates until documented as completed in [release readiness](docs/engineering/RELEASE_READINESS.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [docs/RELEASING.md](docs/RELEASING.md).
