"""Bounded one-factor investigation; each child records complete configuration."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cases = {
    "guide_families": ["--guides", "auto_normal", "auto_diag", "auto_lowrank", "auto_fullrank"],
    "lr005": ["--lr", ".005"],
    "lr05": ["--lr", ".05"],
    "init05": ["--init-scale", ".5"],
    "standardized": ["--features", "standardized"],
    "whitened": ["--features", "whitened"],
    "float64": ["--dtype", "float64"],
    "prior025": ["--prior-sd", ".25"],
    "prior1": ["--prior-sd", "1.0"],
}
for name, flags in cases.items():
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/svi_gaussian_benchmark.py"),
            "--source",
            str(root),
            "--output",
            str(root / f"validation/svi/{name}.json"),
            *flags,
        ],
        check=True,
    )
