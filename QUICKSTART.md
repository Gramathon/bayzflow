# Bayzflow quickstart

Install Bayzflow and choose a Pyro guide:

```bash
python -m pip install bayzflow
bayzflow guide list
bayzflow guide inspect auto_lowrank
```

For a supported PyTorch model with `{"x": tensor, "y": tensor}` batches:

```python
from bayzflow import BayzFlow

predict = lambda model, batch: model(batch["x"])
loss = lambda logits, batch: loss_fn(logits, batch["y"])

bf = BayzFlow.from_model_and_data(
    model=model,
    calib_loader=calibration_loader,
    calib_forward_fn=predict,
    bayes_patterns=["head"],
    guide="auto_lowrank",
    guide_kwargs={"rank": 8},
)
bf.train_posterior(
    train_loader=train_loader,
    loss_from_logits_fn=loss,
    predict_fn=predict,
    num_epochs=10,
    seed=17,
    save_dir="artifacts/posterior",
)

prediction = bf.predict(batch, predict_fn=predict, num_samples=32, return_samples=True)
posterior_sites = bf.posterior_samples(batch, num_samples=32)
summaries = bf.posterior_summary(posterior_sites)
```

Supported automatic Bayesian layer wrappers currently cover Linear, Conv1d/2d/3d and ConvTranspose3d modules. Check `bf.list_candidates()` before selecting patterns. Normalization and arbitrary composite modules are not automatically converted.

## Experiment YAML

Each experiment can specify:

```yaml
guide:
  kind: auto_lowrank
  kwargs:
    rank: 8
```

`BayzFlow.exp("bayzflow/bayzflow.yaml", exp="name")` loads the configured model/data workflow, uses that guide and returns the engine and loaders. The example runner currently trains the configured epoch count; validation-based early stopping is not implemented.

## Custom guides and agents

Register custom Pyro guide factories from trusted application Python with `GuideRegistry`; see [guide docs](docs/guides.md). Install the packaged agent workflow using `bayzflow skill install`. For local MCP discovery, install `bayzflow[mcp]` and run `bayzflow-mcp --stdio`.

The 0.2.0 candidate MCP server supports read-only discovery and guide validation. It does not launch training, sampling, measurement or artifact jobs. See the [README](README.md) for scientific and security limitations.
