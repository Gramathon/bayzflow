# Install Bayzflow

Bayzflow requires Python 3.10 or newer. Use the wheel from the [GitHub release](https://github.com/Gramathon/bayzflow/releases/tag/v0.2.0):

```bash
wheel="https://github.com/Gramathon/bayzflow/releases/download/v0.2.0/bayzflow-0.2.0-py3-none-any.whl"
python -m pip install "$wheel"
```

Select optional dependencies when installing the same wheel. For example:

```bash
python -m pip install "bayzflow[mcp] @ $wheel"
```

Available extras include `mcp` for the official MCP SDK, `fx` for finance examples, `medical` for MONAI and nibabel, `viz` for matplotlib, and `dev` for development tools. Combine extras with commas inside the brackets.

Check the installed package and guide registry:

```bash
bayzflow --version
bayzflow doctor
bayzflow guide list
bayzflow skill status
```

Run local MCP discovery with `bayzflow-mcp --stdio`, or use `bayzflow-mcp --stdio --demo --workspace ~/.local/state/bayzflow-demo` for bounded inference on Linux after installing the `mcp` extra. See the repository [README](README.md) for inference and security scope. GPU support depends on the PyTorch build and has not been validated by the 0.2.0 release checks.
