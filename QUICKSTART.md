# Bayzflow quickstart

[Install Bayzflow from GitHub Releases](INSTALL.md) and inspect the guide registry:

```bash
bayzflow guide list
bayzflow guide inspect auto_lowrank
```

The [README quickstart](README.md#quickstart) is a runnable example. It fits a guide, makes predictions, and writes and reads a checksummed tensor artifact.

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

Register custom Pyro guide factories from trusted application Python with `GuideRegistry`; see [guide docs](docs/guides.md). Install the packaged agent workflow using `bayzflow skill install`. Install the release wheel with the `mcp` extra to use stdio discovery and bounded inference jobs on Linux. See the [MCP guide](docs/MCP.md) for setup and the [README](README.md) for scientific and security limitations.
