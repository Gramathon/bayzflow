"""MCP v2 stdio server. Runs approved jobs in isolated local CPU processes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .jobs import JobStore, Limits
from .service import BayzflowService


def create_server(service: BayzflowService | None = None):
    try:
        from mcp.server import MCPServer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("install Bayzflow with the [mcp] extra to run the MCP server") from exc

    service = service or BayzflowService()
    server = MCPServer(
        "bayzflow",
        instructions="Discover trusted registrations and submit bounded CPU inference jobs.",
    )

    @server.tool()
    def bayzflow_doctor() -> dict[str, Any]:
        """Report Bayzflow and local runtime capabilities."""
        return service.doctor()

    @server.tool()
    def model_list() -> list[dict]:
        """List models explicitly registered by trusted host code."""
        return service.model_list()

    @server.tool()
    def model_inspect(model_id: str) -> dict[str, Any]:
        """Inspect a locally registered model without training it."""
        return service.model_inspect(model_id)

    @server.tool()
    def layer_candidates(model_id: str) -> list[dict]:
        """List supported Bayesian layer candidates for a registered model."""
        return service.layer_candidates(model_id)

    @server.tool()
    def guide_list() -> list[dict]:
        """List built-in and trusted locally registered guide factories."""
        return service.guide_list()

    @server.tool()
    def guide_schema(kind: str) -> dict[str, Any]:
        """Inspect accepted configuration for a registered guide."""
        return service.guide_schema(kind)

    @server.tool()
    def guide_validate(kind: str, kwargs: dict) -> dict[str, Any]:
        """Validate guide configuration without constructing or training a model."""
        return service.guide_validate(kind, kwargs)

    def controlled(fn, *args):
        # Exceptions from trusted model and guide code may contain local paths.
        try:
            return {"ok": True, "result": fn(*args)}
        except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
            return {"ok": False, "error": {"code": "request_rejected"}}
        except (OSError, RuntimeError):
            return {"ok": False, "error": {"code": "operation_unavailable"}}

    @server.tool()
    def bayesian_plan(
        model_id: str,
        layers: list[str],
        kind: str,
        kwargs: dict[str, Any],
        epochs: int = 10,
        seed: int = 0,
        prior_scale: float = 1.0,
        lr: float = 0.03,
    ) -> dict[str, Any]:
        """Check a fit without training, and return its approval ID and resource limits."""
        return controlled(
            service.bayesian_plan, model_id, layers, kind, kwargs, epochs, seed, prior_scale, lr
        )

    @server.tool()
    def posterior_fit(
        plan_id: str, request: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        """Acknowledge a plan by submitting its unchanged request and approval ID."""
        return controlled(service.fit, plan_id, request, idempotency_key)

    @server.tool()
    def job_status(job_id: str) -> dict[str, Any]:
        """Poll a durable fit or measurement job by ID."""
        return controlled(lambda: service.require_jobs().status(job_id))

    @server.tool()
    def job_cancel(job_id: str) -> dict[str, Any]:
        """Request cancellation of an active job and poll job_status for completion."""
        return controlled(lambda: service.require_jobs().cancel(job_id))

    @server.tool()
    def posterior_sample(
        checkpoint_id: str,
        samples: int,
        seed: int,
        idempotency_key: str,
        sites: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queue latent sampling from a verified fit checkpoint."""
        return controlled(
            service.analyse, "sample", checkpoint_id, samples, seed, idempotency_key, sites
        )

    @server.tool()
    def measure_epistemic(
        checkpoint_id: str, samples: int, seed: int, idempotency_key: str
    ) -> dict[str, Any]:
        """Queue variance of model predictions across posterior weight draws."""
        return controlled(
            service.analyse, "epistemic", checkpoint_id, samples, seed, idempotency_key
        )

    @server.tool()
    def measure_layers(
        checkpoint_id: str,
        samples: int,
        seed: int,
        idempotency_key: str,
        layers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Queue parameter uncertainty by selected Bayesian layer."""
        return controlled(
            service.analyse, "layers", checkpoint_id, samples, seed, idempotency_key, layers
        )

    @server.tool()
    def measurement_list() -> list[dict]:
        """List measurements and state what their variances represent."""
        return [
            {
                "tool": "measure_epistemic",
                "quantity": "variance of predictions over posterior weights",
            },
            {
                "tool": "measure_layers",
                "quantity": "variance of layer parameters over posterior weights",
            },
        ]

    @server.tool()
    def posterior_inspect(job_id: str) -> dict[str, Any]:
        """Read verified checkpoint metadata by fit job ID."""
        return controlled(service.posterior_inspect, job_id)

    @server.tool()
    def artifact_list(job_id: str) -> dict[str, Any]:
        """List verified artifact names, sizes, identifiers and SHA256 values."""
        return controlled(lambda: service.require_jobs().status(job_id)["artifacts"])

    @server.tool()
    def artifact_read(
        job_id: str, artifact_id: str, offset: int = 0, length: int = 32768
    ) -> dict[str, Any]:
        """Read one bounded base64 page from a verified artifact."""
        return controlled(lambda: service.require_jobs().read(job_id, artifact_id, offset, length))

    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bayzflow-mcp")
    parser.add_argument(
        "--stdio", action="store_true", help="serve on stdio (the only supported transport)"
    )
    parser.add_argument("--workspace", type=Path, help="private operator-selected job directory")
    parser.add_argument(
        "--demo", action="store_true", help="register the synthetic regression workload"
    )
    args = parser.parse_args(argv)
    service = BayzflowService(
        workspace=JobStore(args.workspace, Limits()) if args.workspace else None
    )
    if args.demo:
        from .workloads import REGRESSION, regression_model

        service.register_model("regression", regression_model(), REGRESSION)
    create_server(service).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
