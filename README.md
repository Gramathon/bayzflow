# Bayzflow

Bayzflow is a Bayesian deep learning framework for selective Bayesian inference and uncertainty analysis in PyTorch models. It uses Pyro variational inference to place posterior distributions over selected supported layers.

The project remains research software. Layer replacement, prior calibration and variational inference are model dependent. Predictive variation is not automatically calibrated uncertainty, and broad architecture compatibility has not been established.

## Install

Python 3.10 or newer is required. Install the wheel from the [GitHub release](https://github.com/Gramathon/bayzflow/releases/tag/v0.2.0):

```bash
python -m pip install "https://github.com/Gramathon/bayzflow/releases/download/v0.2.0/bayzflow-0.2.0-py3-none-any.whl"
```

To include MCP, install the same release wheel with the `mcp` extra:

```bash
python -m pip install "bayzflow[mcp] @ https://github.com/Gramathon/bayzflow/releases/download/v0.2.0/bayzflow-0.2.0-py3-none-any.whl"
```

See [INSTALL.md](INSTALL.md) for the other extras.

## Quickstart

Fit a variational guide to a small regression model and save its posterior predictive samples:

```python
import pyro
import torch
from torch import nn
from torch.utils.data import DataLoader

from bayzflow import BayzFlow

seed = 17
pyro.set_rng_seed(seed)
model = nn.Sequential()
model.add_module("head", nn.Linear(2, 1))
x = torch.linspace(-1, 1, 16).reshape(8, 2)
y = x[:, :1] + 0.2
loader = DataLoader([{"x": row, "y": target} for row, target in zip(x, y)], batch_size=4)
batch = next(iter(loader))


def predict(model, batch):
    return model(batch["x"])


def loss(logits, batch):
    return nn.functional.mse_loss(logits, batch["y"])


bf = BayzFlow.from_model_and_data(
    model=model,
    calib_loader=loader,
    calib_forward_fn=predict,
    bayes_patterns=["head"],
    guide="auto_lowrank",
    guide_kwargs={"rank": 2},
)
bf.train_posterior(
    train_loader=loader,
    loss_from_logits_fn=loss,
    predict_fn=predict,
    num_epochs=10,
    seed=seed,
    save_dir="artifacts/posterior",
)

pyro.set_rng_seed(seed)
result = bf.predict(batch, predict_fn=predict, num_samples=32, return_samples=True)
manifest = bf.save_posterior_samples(
    {"logits": result["samples"]},
    "artifacts/predictive-samples.pt",
    random_seed=seed,
)
reloaded = BayzFlow.load_posterior_sample_artifact("artifacts/predictive-samples.pt.json")
assert torch.equal(reloaded["logits"], result["samples"])
print(reloaded["logits"].shape)
```

Candidate layer wrappers currently cover Linear, Conv1d/2d/3d, and ConvTranspose3d modules. Normalization, ConvTranspose1d/2d and arbitrary composite modules are not candidates. Inspect candidates and verify model behavior on a small dataset before scaling an inference run.

## Guides

Built-in guides are `auto_normal`, `auto_diag`, `auto_lowrank`, and `auto_fullrank`. A JSON-serializable `GuideSpec` can also be passed as `guide=GuideSpec(kind="auto_lowrank", kwargs={"rank": 8})`. Legacy `guide_kind="low_rank"` and `guide_kind="mean_field"` spellings remain accepted by `setup_svi`.

Custom Pyro guide factories are registered by trusted Python code through `GuideRegistry.register`. Serialized configuration selects a local registration; it cannot provide Python code or an import path. See [custom guide notes](docs/guides.md).

## CLI and MCP inference

`bayzflow --version`, `bayzflow doctor`, `bayzflow guide list`, `bayzflow guide inspect auto_lowrank`, and `bayzflow guide validate guide.yaml` provide diagnostics and guide discovery. The packaged skill is available through `bayzflow skill status` and `bayzflow skill install`.

Install the GitHub Release wheel with the `mcp` extra and configure your MCP client to launch the following local stdio command on Linux:

```bash
bayzflow-mcp --stdio --demo --workspace ~/.local/state/bayzflow-demo
```

`bayzflow model inspect regression` describes the bundled demonstration model. In MCP, inspect it with `model_inspect` and `layer_candidates`. Request `bayesian_plan` with model `regression`, layers `["0"]`, kind `auto_diag`, and empty kwargs. Review the returned limits and pass its `plan_id` and unchanged `request` to `posterior_fit` with an idempotency key. Poll `job_status` until it reports success. Supply the fit job ID as `checkpoint_id` to sample the posterior or measure parameter and predictive uncertainty. Read tensor artifacts in pages using the IDs and checksums returned by `artifact_list`.

An operator enables operations by selecting a private workspace. Trusted Python hosts can register CPU models, workload adapters, and custom guides before server creation. See [MCP usage and resource limits](docs/MCP.md).

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

Version 0.2.0 is available as an [immutable GitHub release](https://github.com/Gramathon/bayzflow/releases/tag/v0.2.0). Its validation covers Linux CPU inference and MCP over stdio. See [release readiness](docs/engineering/RELEASE_READINESS.md) for the evidence and scope.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [docs/RELEASING.md](docs/RELEASING.md).
