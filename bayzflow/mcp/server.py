"""MCP v2 stdio server. Only registered in-process objects are discoverable."""

from __future__ import annotations

import argparse

from .service import BayzflowService


def create_server(service: BayzflowService | None = None):
    try:
        from mcp.server import MCPServer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("install Bayzflow with the [mcp] extra to run the MCP server") from exc

    service = service or BayzflowService()
    server = MCPServer("bayzflow", instructions="Read-only discovery for trusted Bayzflow registrations.")

    @server.tool()
    def bayzflow_doctor() -> dict:
        """Report Bayzflow and local runtime capabilities."""
        return service.doctor()

    @server.tool()
    def model_list() -> list[dict]:
        """List models explicitly registered by trusted host code."""
        return service.model_list()

    @server.tool()
    def model_inspect(model_id: str) -> dict:
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
    def guide_schema(kind: str) -> dict:
        """Inspect accepted configuration for a registered guide."""
        return service.guide_schema(kind)

    @server.tool()
    def guide_validate(kind: str, kwargs: dict) -> dict:
        """Validate guide configuration without constructing or training a model."""
        return service.guide_validate(kind, kwargs)

    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bayzflow-mcp")
    parser.add_argument("--stdio", action="store_true", help="serve on stdio (the only supported transport)")
    parser.parse_args(argv)
    create_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
