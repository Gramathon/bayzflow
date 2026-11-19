import argparse
import contextlib
import io
import json
import sys

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
args = p.parse_args()
sys.path.insert(0, args.source)
import pyro
import torch

from bayzflow import BayzFlow


def engine():
    torch.manual_seed(1)
    b = BayzFlow(torch.nn.Sequential(torch.nn.Linear(1, 1)), guide="auto_diag")
    b.selected_module_names = ["0"]
    b.build_bayesian_model()
    b.setup_svi(lambda p, d: ((p - d[1]) ** 2).sum(), lambda m, d: m(d[0]), lr=0.02)
    return b


batch = (torch.ones(3, 1), torch.ones(3, 1))
a, b = engine(), engine()
with contextlib.redirect_stdout(io.StringIO()):
    pyro.set_rng_seed(2)
    a.fit([batch], num_epochs=10)
    pyro.set_rng_seed(2)
    b.fit([batch], num_epochs=5)
    b.fit([batch], num_epochs=5)
x = torch.tensor([[1.0], [1.0 + 1e-10], [1.0 - 1e-10]], dtype=torch.float64)
r = a.posterior_summary({"w": x})["w"]["covariance"]
pyro.clear_param_store()
external = pyro.param("external", torch.tensor(1.0))
a.predict(batch, lambda m, d: m(d[0]), num_samples=2)
print(
    json.dumps(
        {
            "split_parameter_max_delta": max(
                float((v - b._param_store_state["params"][k]).abs().max())
                for k, v in a._param_store_state["params"].items()
            ),
            "covariance_shape": list(r.shape),
            "covariance_dtype": str(r.dtype),
            "covariance": float(r),
            "external_identity_preserved": pyro.param("external") is external,
        },
        indent=2,
    )
)
