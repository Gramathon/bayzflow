"""Record deterministic input arrays and independently check coordinate equivalence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation/svi"))
from reference import dataset, exact_posterior

records = {}
for file in sorted((ROOT / "validation/svi").glob("*.json")):
    result = json.loads(file.read_text())
    if "rows" not in result or "configuration" not in result:
        continue
    config = result["configuration"]
    rows = {}
    for seed in config["seeds"]:
        dtype = getattr(torch, config["dtype"])
        design, y = dataset(seed, dtype)
        torch.manual_seed(seed + 8900)
        base = torch.nn.Linear(2, 1).to(dtype)
        original_mean = torch.cat([base.weight.detach().flatten(), base.bias.detach()])
        transform = torch.eye(3, dtype=dtype)
        if config["features"] == "standardized":
            transform[:2, :2] = torch.diag(1 / design[:, :2].std(0))
        elif config["features"] == "whitened":
            _, singular, vh = torch.linalg.svd(design[:, :2], full_matrices=False)
            transform[:2, :2] = vh.T @ torch.diag(design.new_tensor(48.0).sqrt() / singular)
        inverse = torch.linalg.solve(transform, torch.eye(3, dtype=dtype))
        prior_mean = inverse @ original_mean
        original_covariance = torch.eye(3, dtype=dtype) * config["prior_sd"] ** 2
        transformed_covariance = inverse @ original_covariance @ inverse.T
        # The actual benchmark uses independent Normal prior buffers.
        diagonal_covariance = torch.diag(transformed_covariance.diag())
        torch.testing.assert_close(
            transformed_covariance, diagonal_covariance, atol=1e-6, rtol=1e-6
        )
        m, c = exact_posterior(
            (design @ transform).double(),
            y.double(),
            prior_mean.double(),
            diagonal_covariance.double(),
            0.25,
        )
        native_m, native_c = exact_posterior(
            design.double(), y.double(), original_mean.double(), original_covariance.double(), 0.25
        )
        t = transform.double()
        torch.testing.assert_close(t @ m, native_m, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(t @ c @ t.T, native_c, atol=2e-6, rtol=2e-6)
        arrays = {
            "design": design,
            "y": y,
            "transform": transform,
            "prior_mean": prior_mean,
            "prior_covariance": diagonal_covariance,
            "exact_native_mean": native_m,
            "exact_native_covariance": native_c,
        }
        rows[str(seed)] = {
            k: {
                "values": v.tolist(),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(v.numpy().tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        }
    records[file.stem] = rows
(ROOT / "validation/svi/condition_inputs.json").write_text(json.dumps(records, indent=2))
