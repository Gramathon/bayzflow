import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer.autoguide import AutoMultivariateNormal
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
        guide = default_registry.create(lambda: None, GuideSpec(name, {"init_scale": 1}))
        assert guide is not None
    with pytest.raises(ValueError, match="unsupported arguments"):
        default_registry.validate(GuideSpec("auto_diag", {"rank": 3}))
    with pytest.raises(ValueError, match="positive integer"):
        default_registry.validate(GuideSpec("auto_lowrank", {"rank": 0}))


@pytest.mark.parametrize("value", ["plain text", [], {}, 1, True])
def test_auto_normal_rejects_non_null_create_plates(value):
    with pytest.raises(ValueError, match="create_plates must be null"):
        default_registry.validate("auto_normal", {"create_plates": value})


def test_auto_normal_allows_null_create_plates():
    guide = default_registry.create(lambda: None, "auto_normal", {"create_plates": None})
    assert guide.create_plates is None


def test_python_setup_rejects_fullrank_for_wide_bayesian_layer():
    model = nn.Sequential(nn.Linear(512, 1))
    flow = BayzFlow(model, guide="auto_fullrank")
    flow.selected_module_names = ["0"]
    flow.build_bayesian_model()
    with pytest.raises(ValueError, match="at most 512 latent coordinates; found 513"):
        flow.setup_svi(loss_from_logits_fn=_loss, predict_fn=_callbacks)
    assert flow.guide is None
    assert flow.svi is None


def test_fullrank_checks_native_pyro_latents_before_covariance_allocation():
    def model():
        for site in ("first", "second"):
            pyro.sample(site, dist.Normal(torch.zeros(257), torch.ones(257)).to_event(1))

    guide = default_registry.create(model, "auto_fullrank")
    for _ in range(2):
        with pytest.raises(ValueError, match="at most 512 latent coordinates; found 514"):
            guide()
        assert not hasattr(guide, "scale_tril")


def test_fullrank_loads_checkpoints_with_the_original_pyro_parameter_names(tmp_path, monkeypatch):
    def make_flow():
        flow = BayzFlow(_TinyModel(), guide="auto_fullrank")
        flow.selected_module_names = ["linear"]
        flow.build_bayesian_model()
        return flow

    loader = _data()
    batch = next(iter(loader))
    with monkeypatch.context() as patch:
        patch.setattr("bayzflow.guides.registry.BoundedFullRankGuide", AutoMultivariateNormal)
        original = make_flow()
        original.train_posterior(loader, _loss, _callbacks, num_epochs=1, save_dir=str(tmp_path))
        expected = original.posterior_covariance(batch)["cholesky"]

    restored = make_flow()
    restored.load_posterior(str(tmp_path), _loss, _callbacks)
    assert torch.equal(restored.posterior_covariance(batch)["cholesky"], expected)


def test_sample_artifact_writer_requires_named_tensors(tmp_path):
    flow = BayzFlow(_TinyModel())
    output = tmp_path / "bad.pt"
    with pytest.raises(ValueError, match="mapping of site names to tensors"):
        flow.save_posterior_samples(torch.zeros(3, 2), str(output))
    assert not output.exists()


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


@pytest.mark.parametrize(("kind", "representation"), [
    ("auto_diag", "diagonal"),
    ("auto_lowrank", "diagonal_plus_factor_product"),
    ("auto_fullrank", "cholesky"),
    ("auto_normal", "unavailable"),
])
def test_parameter_and_covariance_representations(kind, representation, tmp_path):
    from bayzflow.mcp.workloads import (
        regression_batch,
        regression_loss,
        regression_model,
        regression_predict,
    )

    batch = regression_batch()
    kwargs = {"rank": 1} if kind == "auto_lowrank" else {}
    flow = BayzFlow(regression_model(), guide=kind, guide_kwargs=kwargs)
    flow.selected_module_names = ["0"]
    flow.build_bayesian_model()
    flow.train_posterior(DataLoader(TensorDataset(*batch), batch_size=len(batch[0])),
                         regression_loss, regression_predict, num_epochs=1, save_dir=str(tmp_path))
    assert flow.posterior_parameters()
    assert flow.posterior_covariance(batch)["representation"] == representation
    if kind == "auto_lowrank":
        native = flow.posterior_covariance(batch)
        assert native["factor"].shape == (2, 1)
        assert torch.all(native["diagonal"] > 0)


def test_convolutions_have_distinct_posterior_sites():
    model = nn.Sequential(nn.Conv1d(1, 1, 2), nn.Conv1d(1, 1, 2))
    flow = BayzFlow(model, guide="auto_diag")
    flow.selected_module_names = ["0", "1"]
    flow.build_bayesian_model()
    batch = torch.ones((3, 1, 5)), torch.zeros((3, 1, 3))
    flow.train_posterior(DataLoader(TensorDataset(*batch), batch_size=3), _loss, _callbacks,
                         num_epochs=1)
    assert set(flow.posterior_samples(batch, 3)) >= {"0.weight", "1.weight", "0.bias", "1.bias"}
