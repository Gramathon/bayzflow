# Packaging validation

## Build

Command executed from the repository root:

```bash
PYTHONPATH=/tmp/bayzflow-build-tools:$PYTHONPATH python3 -m build
```

Result: built both distributions successfully with isolated setuptools 84.2.1 and wheel 0.48.0. Source distribution and wheel version are 0.2.0. The wheel contains `bayzflow/bayzflow.yaml` and the complete packaged `bayesian-inference` skill. Both distributions exclude unrelated local research, observability and archive trees. The sdist omits internal `docs/engineering` validation reports so the recorded archive checksum remains reproducible.

Artifacts:

| File | SHA256 |
|---|---|
| `bayzflow-0.2.0.tar.gz` | `33e1af86420b47c9e2793425b1bf0ac1efff9cc78f4f6c3d4b24ad1f110e387f` |
| `bayzflow-0.2.0-py3-none-any.whl` | `18c363d57cb1fab6ba51ea28151759d33d979a68ff2c6ecae50be581a84fd533` |

## Installed-wheel checks

Created `/tmp/bayzflow-clean-venv` with Python 3.10.12 and host system site packages enabled, then force-installed the wheel with `pip install --no-deps`. This validates installed package/resource resolution outside the checkout, while reusing the host's PyTorch 2.4.1+cu121 and Pyro 1.9.1 rather than resolving all dependencies into the new environment.

From `/tmp`, the installed commands returned version 0.2.0, printed doctor details, listed all four guides, reported the skill packaged at the venv site-packages resource path, and installed all skill files to `/tmp/bayzflow-installed-skill-final`. `bayzflow.__file__` resolved to `/tmp/bayzflow-clean-venv/lib/python3.10/site-packages/bayzflow/__init__.py` when run outside the source checkout. A one-epoch tiny CPU inference using the installed wheel trained with `auto_diag`, sampled posterior sites and saved/reloaded a checksummed tensor artifact.

The installed `bayzflow-mcp --stdio` entry point was launched as a subprocess through the official MCP 2.3.0 client. It listed seven tools and returned a successful `guide_validate` response.

## Limits

This is a local build/install check on Python 3.10 only. No Python 3.11/3.12 matrix, platform matrix, independent dependency resolution, PyPI installation, or GitHub release URL installation was exercised. The `python -m build` output includes setuptools warnings about unmatched recursive YAML and prune patterns; the wheel and sdist content checks nevertheless confirmed the required YAML and skill assets and excluded user research artifacts.
