"""Validate retained comparisons and write compact, reproducible numerical tables."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "validation/svi"
summary = {}
for path in sorted(results.glob("*.json")):
    data = json.loads(path.read_text())
    if "rows" not in data or "configuration" not in data:
        continue
    grouped = {}
    parity = 0.0
    for row in data["rows"]:
        key = row["seed"], row["guide"]
        grouped.setdefault(key, {})[row["implementation"]] = row
    for pair in grouped.values():
        for step in pair["direct"]["curve"]:
            for metric in [
                "mean_error",
                "relative_covariance_error",
                "gaussian_kl",
                "predictive_variance_error",
            ]:
                parity = max(
                    parity,
                    abs(
                        pair["direct"]["curve"][step][metric]
                        - pair["bayzflow"]["curve"][step][metric]
                    ),
                )
    assert parity < 1e-5, (path, parity)
    summaries = {}
    for kind in sorted({r["guide"] for r in data["rows"]}):
        selected = [
            r for r in data["rows"] if r["implementation"] == "bayzflow" and r["guide"] == kind
        ]
        final = [r["curve"][str(data["configuration"]["steps"])] for r in selected]
        summaries[kind] = {
            metric: [r[metric] for r in final]
            for metric in [
                "mean_error",
                "relative_covariance_error",
                "gaussian_kl",
                "predictive_variance_error",
                "trace",
                "eigenvalues",
                "leading_subspace_agreement",
            ]
        }
        summaries[kind]["runtime_seconds"] = [r["seconds"] for r in selected]
    summary[path.stem] = {"max_parity_discrepancy": parity, "guides": summaries}
(results / "summary.json").write_text(json.dumps(summary, indent=2))
lines = [
    "| Condition | Guide | Mean error median | Relative covariance errors (11 / 23 / 37) | KL median | Predictive variance error median |",
    "|---|---|---:|---|---:|---:|",
]
for condition, value in summary.items():
    for kind, metrics in value["guides"].items():
        lines.append(
            f"| {condition} | {kind} | {statistics.median(metrics['mean_error']):.4f} | "
            + " / ".join(f"{x:.4f}" for x in metrics["relative_covariance_error"])
            + f" | {statistics.median(metrics['gaussian_kl']):.4f} | {statistics.median(metrics['predictive_variance_error']):.4f} |"
        )
(results / "tables.md").write_text("\n".join(lines) + "\n")
checksums = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(results.iterdir())
    if p.is_file() and p.name != "checksums.json"
}
(results / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
