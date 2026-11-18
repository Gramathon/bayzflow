# Variational guides

`GuideSpec` is the canonical, JSON-serializable selection object:

```python
from bayzflow import BayzFlow
from bayzflow.guides import GuideSpec

engine = BayzFlow(model, guide=GuideSpec("auto_lowrank", {"rank": 8}))
```

The built-in identifiers map to Pyro `AutoNormal`, `AutoDiagonalNormal`, `AutoLowRankMultivariateNormal`, and `AutoMultivariateNormal`. Constructor keyword names are checked against the installed Pyro classes. `auto_lowrank` retains the legacy explicit rank 20 default; configure `rank` for a smaller latent model. The engine preserves `guide_kind="low_rank"` and `guide_kind="mean_field"` aliases for existing callers.

Trusted application code may register a real Pyro guide factory:

```python
from dataclasses import dataclass
from pyro.infer.autoguide import AutoDiagonalNormal
from bayzflow.guides import GuideRegistry

@dataclass
class DiagonalConfig:
    init_scale: float = 0.1

def make_guide(model, init_scale=0.1):
    return AutoDiagonalNormal(model, init_scale=init_scale)

registry = GuideRegistry()
registry.register("diagonal_v1", make_guide, config_schema=DiagonalConfig,
                  version="1.0", capabilities={"covariance": False})
engine = BayzFlow(model, guide="custom:diagonal_v1", guide_registry=registry)
```

The factory identity and version become part of guide metadata. Registration is a trusted-code operation. A serialized spec cannot load a module or function. Guide compatibility still depends on the actual model trace and latent dimension; successful construction is not proof of convergence or calibration.
