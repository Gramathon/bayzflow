# BayzFlow Installation Guide

## Quick Installation

### For Development (Editable Install)

```bash
# Clone or navigate to the repository
cd bayzflow

# Install in editable mode with basic dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### From Source

```bash
git clone <repository-url>
cd bayzflow
pip install .
```

## Optional Dependencies

### Financial Examples (FX Trading)

```bash
pip install -e ".[fx]"
```

Includes: yfinance, pandas, ta (technical analysis)

### Medical Imaging (MONAI)

```bash
pip install -e ".[medical]"
```

Includes: monai, nibabel

### Visualization

```bash
pip install -e ".[viz]"
```

Includes: matplotlib

### Development Tools

```bash
pip install -e ".[dev]"
```

Includes: pytest, black, isort, flake8

### All Optional Dependencies

```bash
pip install -e ".[all]"
```

## System Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended for medical imaging examples)
- 8GB+ RAM (16GB+ recommended for medical imaging)

## Core Dependencies

The following are automatically installed:

- torch >= 1.13.0
- pyro-ppl >= 1.8.0
- numpy >= 1.21.0
- pyyaml >= 5.4.0

## Verify Installation

```bash
python -c "from bayzflow import BayzFlow, __version__; print(f'BayzFlow {__version__} installed successfully')"
```

## Running Examples

### FX LSTM Example

```bash
python bayzflow/examples/fx/fx_lstm_train.py
```

### MONAI U-Net Example

```bash
# Requires: pip install -e ".[medical]"
python bayzflow/examples/monai_examples/monai_unet_train.py
```

### SwinUNETR Example (Pretrained Encoder)

```bash
# Edit bayzflow/bayzflow.yaml to set default_experiment: "monai_swin_unetr_pretrained"
python bayzflow/examples/monai_examples/monai_unet_train.py
```

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA-related errors:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install specific CUDA version of PyTorch if needed
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### MONAI Installation

If MONAI installation fails:

```bash
# Try installing MONAI separately
pip install 'monai[all]'
```

### Import Errors

If you encounter import errors after installation:

```bash
# Reinstall in editable mode
pip uninstall bayzflow
pip install -e .
```

## Uninstall

```bash
pip uninstall bayzflow
```

## Building Distribution Packages

```bash
# Install build tools
pip install build

# Build wheel and source distribution
python -m build

# Install from wheel
pip install dist/bayzflow-0.1.0-py3-none-any.whl
```
