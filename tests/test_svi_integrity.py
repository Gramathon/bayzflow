"""Deterministic lifecycle and representation regression tests."""

import sys
from pathlib import Path

import pyro
import pytest
import torch

from bayzflow import BayzFlow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "svi"))
from reference import dataset, direct_model, exact_posterior, fidelity, nll


def make_engine(policy="joint", kind="auto_diag"):
    torch.manual_seed(43)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1))
    bf = BayzFlow(
        model,
        guide=kind,
        guide_kwargs={"rank": 1} if kind == "auto_lowrank" else {},
        deterministic_update_policy=policy,
    )
    bf.selected_module_names = ["2"]
    bf.build_bayesian_model()
    bf.setup_svi(nll, lambda m, b: m(b[0]), lr=0.02)
    x, y = dataset(11)
    return bf, (x[:, :2], y[:, None])


def test_continuation_matches_uninterrupted_and_preserves_store_identity():
    a, batch = make_engine()
    b, _ = make_engine()
    pyro.set_rng_seed(71)
    a.fit([batch], num_epochs=10)
    pyro.set_rng_seed(71)
    b.fit([batch], num_epochs=5)
    identities = {k: id(v) for k, v in b._param_store_state["params"].items()}
    b.fit([batch], num_epochs=5, continue_optimizer=True)
    for name, value in a._param_store_state["params"].items():
        torch.testing.assert_close(value, b._param_store_state["params"][name], rtol=0, atol=0)
        assert identities[name] == id(b._param_store_state["params"][name])
    assert b.training_history == a.training_history


@pytest.mark.parametrize("policy", ["freeze", "joint"])
def test_training_policy_and_updates(policy):
    bf, batch = make_engine(policy)
    before = {k: v.detach().clone() for k, v in bf.bayes_model.named_parameters()}
    bf.fit([batch], num_epochs=8)
    deltas = [not torch.equal(before[k], v) for k, v in bf.bayes_model.named_parameters()]
    assert any(deltas) if policy == "joint" else not any(deltas)
    report = bf.diagnostics(batch)
    assert report["training"]["deterministic_update_policy"] == policy
    assert report["training"]["svi_steps"] == 8
    assert report["parameter_updates"]


def test_explicit_reset_and_restart():
    bf, batch = make_engine()
    bf.fit([batch], num_epochs=3)
    params = {k: v.clone() for k, v in bf._param_store_state["params"].items()}
    bf.reset_optimizer()
    assert bf.optimizer.get_state() == {}
    assert all(torch.equal(v, bf._param_store_state["params"][k]) for k, v in params.items())
    bf.setup_svi(nll, lambda m, b: m(b[0]), restart_posterior=True)
    assert bf._param_store_state["params"] == {}
    assert bf.training_history == []


def test_summary_preserves_precision_and_matrix_shape():
    bf, _ = make_engine()
    values = torch.tensor([[1.0], [1.0 + 1e-10], [1.0 - 1e-10]], dtype=torch.float64)
    result = bf.posterior_summary({"w": values})["w"]
    assert result["covariance"].shape == (1, 1)
    assert result["covariance"].dtype == torch.float64
    assert result["covariance"][0, 0] > 0
    with pytest.raises(ValueError, match="finite"):
        bf.posterior_summary({"w": values * float("nan")})


def test_exact_gaussian_and_coordinate_transform():
    x = torch.eye(2, dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    m, c = exact_posterior(x, torch.ones(2, dtype=torch.float64), z, x, 1.0)
    torch.testing.assert_close(m, torch.full_like(m, 0.5))
    torch.testing.assert_close(c, x * 0.5)
    t = torch.tensor([[2.0, 1.0], [0.0, 0.5]], dtype=torch.float64)
    inverse = torch.linalg.solve(t, x)
    mt, ct = exact_posterior(
        x @ t, torch.ones(2, dtype=torch.float64), inverse @ z, inverse @ x @ inverse.T, 1.0
    )
    torch.testing.assert_close(t @ mt, m)
    torch.testing.assert_close(t @ ct @ t.T, c)
    assert abs(fidelity(m, c, m, c, x)["gaussian_kl"]) < 1e-12


@pytest.mark.parametrize("kind", ["auto_normal", "auto_diag", "auto_lowrank", "auto_fullrank"])
def test_native_guide_moments_and_sampling(kind):
    bf, batch = make_engine(kind=kind)
    bf.fit([batch], num_epochs=2)
    report = bf.diagnostics(batch)
    posterior = report["posterior"]
    assert posterior["coordinate_system"] == "unconstrained_latent"
    samples = bf.posterior_samples(
        batch, num_samples=3000, max_samples=3000, site_names=["2.weight", "2.bias"]
    )
    draws = torch.cat(
        [samples[k].reshape(3000, -1) for k in ["2.weight", "2.bias"]], dim=1
    ).double()
    mean = torch.tensor(posterior["mean"], dtype=torch.float64)
    cov = torch.tensor(posterior["covariance"], dtype=torch.float64)
    assert (draws.mean(0) - mean).norm() / cov.trace().sqrt() < 0.08
    assert (torch.cov(draws.T) - cov).norm() / cov.norm() < 0.12
    with pytest.raises(ValueError, match="dimension"):
        bf.diagnostics(batch, max_dimension=1)


def test_external_store_identity_is_preserved():
    pyro.clear_param_store()
    external = pyro.param("external", torch.tensor(1.0))
    bf, batch = make_engine()
    bf.fit([batch], num_epochs=1)
    assert pyro.param("external") is external
    pyro.clear_param_store()


def test_objective_and_gradient_parity():
    from pyro import poutine

    x, y = dataset(11, torch.float64)
    batch = (x[:, :2], y[:, None])
    mean = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    base = torch.nn.Sequential(torch.nn.Linear(2, 1)).double()
    base[0].weight.data.copy_(mean[:2].reshape(1, 2))
    base[0].bias.data.copy_(mean[2:])
    bf = BayzFlow(base, guide="auto_fullrank")
    bf.selected_module_names = ["0"]
    bf.build_bayesian_model(default_prior_scale=0.5)
    bf.setup_svi(nll, lambda m, b: m(b[0]))
    w = torch.tensor([[0.3, -0.5]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    sites = {"0.weight": w, "0.bias": b}
    values, gradients = [], []
    for model in [direct_model(mean, torch.full_like(mean, 0.5)), bf._pyro_model]:
        trace = poutine.trace(poutine.condition(model, data=sites)).get_trace(batch)
        value = trace.log_prob_sum()
        values.append(value)
        gradients.append(torch.cat([g.flatten() for g in torch.autograd.grad(value, (w, b))]))
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=1e-12)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=1e-12)


def test_nonfinite_loss_and_policy_validation():
    with pytest.raises(ValueError, match="policy"):
        BayzFlow(torch.nn.Linear(1, 1), deterministic_update_policy="implicit")
    bf, batch = make_engine()
    bf.setup_svi(lambda p, b: p.sum() * float("nan"), lambda m, b: m(b[0]))
    with pytest.raises((FloatingPointError, ValueError)):
        bf.fit([batch], num_epochs=1)


def test_guide_updates_are_observed_and_legacy_reset_is_explicit():
    bf, batch = make_engine()
    bf.fit([batch], num_epochs=2)
    before = {n: p.detach().clone() for n, p in bf._param_store_state["params"].items()}
    bf.fit([batch], num_epochs=1)  # legacy optimizer reset
    steps = [float(v["state"][0]["step"]) for v in bf.optimizer.get_state().values()]
    assert steps and set(steps) == {1.0}
    assert any(not torch.equal(v, bf._param_store_state["params"][n]) for n, v in before.items())
    assert any(v and v > 0 for v in bf.guide_parameter_updates.values())


def test_checkpoint_load_declares_optimizer_limitation(tmp_path):
    bf, batch = make_engine("freeze")
    bf.train_posterior([batch], nll, lambda m, b: m(b[0]), num_epochs=2, save_dir=str(tmp_path))
    restored, _ = make_engine("freeze")
    restored.load_posterior(str(tmp_path), nll, lambda m, b: m(b[0]))
    with pytest.raises(ValueError, match="no optimizer state"):
        restored.fit([batch], num_epochs=1, continue_optimizer=True)
    restored.fit([batch], num_epochs=1, continue_optimizer=False)
    other, _ = make_engine("joint")
    with pytest.raises(ValueError, match="policy mismatch"):
        other.load_posterior(str(tmp_path), nll, lambda m, b: m(b[0]))


def test_diagnostics_preserve_rng_and_freeze_detects_tampering():
    bf, batch = make_engine("freeze")
    bf.fit([batch], num_epochs=1)
    rng = torch.get_rng_state().clone()
    bf.diagnostics(batch)
    assert torch.equal(torch.get_rng_state(), rng)
    next(bf.bayes_model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="freeze policy violated"):
        bf.fit([batch], num_epochs=1)


def test_diagnostics_bound_precedes_fullrank_initialization():
    bf, batch = make_engine(kind="auto_fullrank")
    assert bf.guide.prototype_trace is None
    with pytest.raises(ValueError, match="dimension"):
        bf.diagnostics(batch, max_dimension=1)
    assert bf.guide.prototype_trace is None


def test_cross_engine_recursive_reentry_fails_without_store_leak():
    from bayzflow.core.engine import _engine_param_store

    a, _ = make_engine()
    b, _ = make_engine()
    with _engine_param_store(a):
        marker = pyro.param("engine_a_marker", torch.tensor(1.0))
        with _engine_param_store(b):
            assert "engine_a_marker" not in pyro.get_param_store()
            with pytest.raises(RuntimeError, match="reenter"), _engine_param_store(a):
                pass
        assert pyro.param("engine_a_marker") is marker
