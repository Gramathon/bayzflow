# Install Bayzflow

Bayzflow requires Python 3.10 or newer. Install the core package from the source checkout or a built wheel:

```bash
python -m pip install .
```

Optional groups are independently installable:

```bash
python -m pip install 'bayzflow[mcp]'      # official MCP SDK v2 server
python -m pip install 'bayzflow[fx]'       # finance example dependencies
python -m pip install 'bayzflow[medical]'  # MONAI and nibabel
python -m pip install 'bayzflow[viz]'      # matplotlib
python -m pip install 'bayzflow[dev]'      # pytest, Ruff and Mypy
```

Check the installed package and guide registry:

```bash
bayzflow --version
bayzflow doctor
bayzflow guide list
bayzflow skill status
```

Run the local discovery-only MCP service with `bayzflow-mcp --stdio` after installing the `mcp` extra. See the repository [README](README.md) for inference and security scope. GPU support depends on the PyTorch build and has not been validated by the 0.2.0 candidate checks.
