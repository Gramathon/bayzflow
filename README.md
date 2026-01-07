# BayzFlow

BayzFlow is a Bayesian deep learning framework that converts PyTorch models into
calibrated Bayesian (Pyro) models with uncertainty quantification.

## Features

- Works with any `torch.nn.Module`.
- Pattern-based selection of layers to bayesianize.
- Automatic prior calibration from data.
- Monte Carlo prediction with uncertainty estimates.

## Installation

- Requires Python 3.8+, PyTorch >= 1.13, and Pyro >= 1.8.
- Install in editable mode: `pip install -e .`
- Optional extras:
  - `pip install -e ".[fx]"`
  - `pip install -e ".[medical]"`
  - `pip install -e ".[viz]"`
  - `pip install -e ".[all]"`

## Quickstart

Data loaders should yield dict batches like `{"x": tensor, "y": tensor}`.

```python
from bayzflow import BayzFlow

bf = BayzFlow.auto(
    model=model,
    calib_loader=calib_loader,
    train_loader=train_loader,
    calib_forward_fn=lambda m, b: m(b["x"]),
    loss_fn=loss_fn,
    bayes_patterns=["head"],
    train_epochs=10,
)

preds = bf.predict(
    batch=test_batch,
    predict_fn=lambda m, b: m(b["x"]),
    num_samples=16,
)
```

## Configuration

Experiment presets live in `bayzflow/bayzflow.yaml`.

```python
bf, loaders, cfg, extras = BayzFlow.exp(
    "bayzflow/bayzflow.yaml",
    exp="monai_unet_3d",
)
```

## Examples

- `python bayzflow/examples/fx/fx_lstm_train.py`
- `python bayzflow/examples/monai_examples/monai_unet_train.py`
- `python bayzflow/examples/synthetic3d.py`

## Repository Layout

- `bayzflow/core/` engine and experiment runner.
- `bayzflow/models/` model wrappers.
- `bayzflow/examples/` runnable scripts.
- `bayzflow/bayzflow.yaml` experiment configuration.

## Development

- `pip install -e ".[dev]"`
- Format: `black .` and `isort .`
- Lint: `flake8`
- Tests: `pytest` (when tests are present)

## License

MIT
