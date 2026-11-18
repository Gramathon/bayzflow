import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bayzflow import BayzFlow
from bayzflow.core.engine import attach_priors, find_candidate_modules
from bayzflow.guides import GuideRegistry, GuideSpec, default_registry
from bayzflow.models.blstm import BayesianLSTM


def test_guide_spec_round_trip_and_identity():
    spec = GuideSpec("auto_lowrank", {"rank": 2})
    assert GuideSpec.from_dict(spec.to_dict()) == spec
    assert spec.identity == GuideSpec("auto_lowrank", {"rank": 2}).identity
    with pytest.raises(ValueError):
        GuideSpec("auto_diag", {"callable": object()})


def test_builtins_validate_constructor_arguments():
    for name in ("auto_normal", "auto_diag", "auto_lowrank", "auto_fullrank"):
        assert default_registry.validate(GuideSpec(name))
    with pytest.raises(ValueError, match="unsupported arguments"):
        default_registry.validate(GuideSpec("auto_diag", {"rank": 3}))
    with pytest.raises(ValueError, match="positive integer"):
        default_registry.validate(GuideSpec("auto_lowrank", {"rank": 0}))


def test_unsupported_module_types_are_not_candidates_and_fail_closed():
    model = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 1))
    assert [item.name for item in find_candidate_modules(model)] == ["1"]
    with pytest.raises(TypeError, match="unsupported type"):
        attach_priors(model, ["0"])


def _custom_guide(model, init_scale=0.1):
    from pyro.infer.autoguide import AutoDiagonalNormal

    return AutoDiagonalNormal(model, init_scale=init_scale)


@dataclass
class _CustomGuideConfig:
    init_scale: float = 0.1


def _data():
    torch.manual_seed(19)
    x = torch.linspace(-1, 1, 16).unsqueeze(-1)
    y = 1.5 * x + 0.2
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return loader


def _callbacks(model, batch):
    x, _ = batch
    return model(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def _loss(logits, batch):
    _, y = batch
    return ((logits - y) ** 2).mean()


def test_guide_registry_and_custom_factory_end_to_end(tmp_path):
    registry = GuideRegistry()
    registry.register("local_diag", _custom_guide, config_schema=_CustomGuideConfig,
                      version="1.2", capabilities={"covariance": False})
    spec = registry.validate(GuideSpec("custom:local_diag", {"init_scale": 0.15}))
    assert spec.implementation_version == "1.2"
    with pytest.raises(ValueError, match="must be float"):
        registry.validate(GuideSpec("custom:local_diag", {"init_scale": "invalid"}))

    registry.register("test_local_diag", _custom_guide, config_schema=_CustomGuideConfig, version="1.2")
    loader = _data()
    model = _TinyModel()
    bf = BayzFlow.from_model_and_data(
        model=model,
        calib_loader=loader,
        calib_forward_fn=_callbacks,
        bayes_patterns=["linear"],
        num_calib_passes=1,
        guide="custom:test_local_diag",
        guide_kwargs={"init_scale": 0.15},
        guide_registry=registry,
        verbose=False,
    )
    bf.train_posterior(loader, _loss, _callbacks, num_epochs=2, save_dir=str(tmp_path), seed=42)
    assert bf.guide_spec.kind == "custom:test_local_diag"
    assert (tmp_path / "meta.json").exists()
    checkpoint_meta = json.loads((tmp_path / "meta.json").read_text())
    assert checkpoint_meta["guide"]["kind"] == "custom:test_local_diag"
    assert checkpoint_meta["random_seed"] == 42
    assert set(checkpoint_meta["artifact_sha256"]) == {"variational_tensors", "constraints", "model_state"}

    prediction = bf.predict(next(iter(loader)), _callbacks, num_samples=4, return_samples=True)
    assert prediction["samples"].shape[0] == 4
    restored = BayzFlow.from_model_and_data(
        _TinyModel(), loader, _callbacks, ["linear"], num_calib_passes=1,
        guide="auto_diag", guide_registry=registry, verbose=False,
    )
    restored.load_posterior(str(tmp_path), _loss, _callbacks)
    assert restored.guide_spec.kind == "custom:test_local_diag"
    for key, expected in bf._param_store_state["params"].items():
        assert torch.equal(expected, restored._param_store_state["params"][key])
    for key, expected in bf.bayes_model.state_dict().items():
        assert torch.equal(expected, restored.bayes_model.state_dict()[key])
    actual = restored.predict(next(iter(loader)), _callbacks, num_samples=2)["mean"]
    assert torch.isfinite(actual).all()
    latent_samples = restored.posterior_samples(next(iter(loader)), num_samples=4)
    assert any(name.endswith("weight") for name in latent_samples)
    summary = restored.posterior_summary(latent_samples)
    assert all(item["sample_count"] == 4 for item in summary.values())
    artifact = restored.save_posterior_samples(
        latent_samples, str(tmp_path / "posterior-samples.pt"), random_seed=42
    )
    assert artifact["checkpoint_identity"] is not None
    assert artifact["random_seed"] == 42
    recovered = BayzFlow.load_posterior_sample_artifact(str(tmp_path / "posterior-samples.pt.json"))
    assert all(torch.equal(latent_samples[name].cpu(), recovered[name]) for name in latent_samples)
    with pytest.raises(ValueError, match="num_samples"):
        restored.posterior_samples(next(iter(loader)), num_samples=5, max_samples=4)

    with (tmp_path / "pyro_params.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        restored.load_posterior(str(tmp_path), _loss, _callbacks)


def test_engine_param_stores_are_isolated_between_instances():
    loader = _data()
    engines = []
    for guide_name in ("auto_diag", "auto_normal"):
        engine = BayzFlow.from_model_and_data(
            _TinyModel(), loader, _callbacks, ["linear"], num_calib_passes=1,
            guide=guide_name, verbose=False,
        )
        engines.append(engine)

    def train_and_predict(engine):
        engine.train_posterior(loader, _loss, _callbacks, num_epochs=1)
        return engine.predict(next(iter(loader)), _callbacks, num_samples=2)["mean"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(train_and_predict, engines))
    assert first.shape == second.shape
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("guide_name", ["auto_normal", "auto_diag", "auto_lowrank", "auto_fullrank"])
def test_builtin_guides_train_and_predict(guide_name):
    loader = _data()
    engine = BayzFlow.from_model_and_data(
        _TinyModel(), loader, _callbacks, ["linear"], num_calib_passes=1,
        guide=guide_name, guide_kwargs={"rank": 1} if guide_name == "auto_lowrank" else {},
        verbose=False,
    )
    engine.train_posterior(loader, _loss, _callbacks, num_epochs=1)
    output = engine.predict(next(iter(loader)), _callbacks, num_samples=3, return_samples=True)
    assert output["samples"].shape[0] == 3
    assert torch.isfinite(output["mean"]).all()


def test_preexisting_pyro_model_trains_without_double_wrapping():
    torch.manual_seed(11)
    model = BayesianLSTM(input_size=2, hidden_size=3)
    x = torch.randn(6, 4, 2)
    y = torch.randn(6)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    def predict_fn(current, batch):
        return current(batch[0], batch[1])

    bf = BayzFlow(model, guide="auto_diag")
    bf.bayes_model = model
    bf.train_posterior(loader, lambda logits, batch: logits.new_zeros(()), predict_fn,
                       num_epochs=1)
    assert isinstance(bf.guide, nn.Module)
    assert torch.isfinite(bf.predict((x[:3], y[:3]), predict_fn, num_samples=2)["mean"]).all()
