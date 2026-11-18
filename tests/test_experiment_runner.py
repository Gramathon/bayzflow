import json
import sys
import types

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from bayzflow.core import experiment
from bayzflow.models.blstm import BayesianLSTM


class _TinyDataset:
    def __init__(self, **kwargs):
        x = torch.randn(8, 3, 1)
        y = torch.randn(8)
        self.loader = DataLoader(TensorDataset(x, y), batch_size=4)

    def train_loader(self, batch_size=4, shuffle=False):
        return self.loader

    def val_loader(self, batch_size=4, shuffle=False):
        return self.loader


def test_exp_runner_passes_yaml_guide_and_uses_engine_signatures(tmp_path, monkeypatch):
    module_name = "bayzflow.examples.fx.fx_utils.fx_dataloader"
    fake_module = types.ModuleType(module_name)
    fake_module.FXDataset = _TinyDataset
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    def make_model(input_size=1, hidden_size=2, **kwargs):
        return BayesianLSTM(input_size=input_size, hidden_size=hidden_size)

    monkeypatch.setattr(experiment, "_resolve_class", lambda _: make_model)
    checkpoint_dir = tmp_path / "posterior"
    config = {
        "bayzflow": {
            "default_experiment": "tiny",
            "experiments": {
                "tiny": {
                    "kind": "fx",
                    "model": {"module": "unused.Model", "input_size": 1, "hidden_size": 2},
                    "data": {"module": "unused.Dataset"},
                    "guide": {"kind": "auto_diag", "kwargs": {}},
                    "training": {"epochs": 1, "batch_size": 4, "seed": 17},
                    "checkpoints": {"outdir": str(checkpoint_dir), "best_name": "unused.pt"},
                }
            },
        }
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    bf, loaders, _cfg, extras = experiment.exp(str(path))
    assert bf.guide_spec.kind == "auto_diag"
    assert bf._param_store_state["params"]
    assert loaders["train"] is not None
    assert extras["checkpoint_path"] == str(checkpoint_dir)
    assert (checkpoint_dir / "meta.json").is_file()
    metadata = json.loads((checkpoint_dir / "meta.json").read_text())
    assert metadata["experiment_metadata"]["model"]["hidden_size"] == 2
    assert len(metadata["training_metrics"]["elbo_per_epoch"]) == 1
    assert yaml.safe_load(path.read_text())["bayzflow"]["experiments"]["tiny"]["guide"]["kind"] == "auto_diag"
