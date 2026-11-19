#!/usr/bin/env bash
# Prepare an isolated, committed EXPCON v0.2.1 project from a Bayzflow wheel.
set -euo pipefail
wheel=$(realpath "${1:?wheel path required}")
target=$(realpath -m "${2:?new project directory required}")
source_dir=$(dirname "$(realpath "$0")")
if [ "$(basename "$wheel")" != bayzflow-0.2.1-py3-none-any.whl ]; then
    echo "Expected a Bayzflow 0.2.1 wheel" >&2
    exit 1
fi
mkdir "$target"
cd "$target"
mkdir -p experiments/gaussian scripts
cp "$source_dir/EXPERIMENT.yaml" experiments/gaussian/
if [ "${3:-v001}" != v001 ]; then
    [[ "$3" =~ ^v[0-9]+$ ]]
    sed -i "s/version: v001/version: $3/" experiments/gaussian/EXPERIMENT.yaml
fi
cp "$source_dir/run.py" scripts/
cp "$wheel" .
cat > pyproject.toml <<'TOML'
[project]
name = "bayzflow-experiment"
version = "0.1.0"
requires-python = ">=3.12.1,<3.13"
dependencies = ["bayzflow==0.2.1", "torch==2.15.0+cpu"]

[tool.uv.sources]
bayzflow = {path = "bayzflow-0.2.1-py3-none-any.whl"}
torch = {index = "pytorch-cpu"}

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
TOML
printf '.venv/\n.expcon/\n' > .gitignore
uv lock --python 3.12
uv sync --frozen --python 3.12 --no-install-project
sha256sum bayzflow-0.2.1-py3-none-any.whl > wheel.sha256
git init -q
git add .
git -c user.name="Bayzflow Example" -c user.email="example@example.invalid" commit -qm "freeze inputs for gaussian regression"
