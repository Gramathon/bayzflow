"""Command line interface for package diagnostics, guides and skill assets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from bayzflow import __version__
from bayzflow.guides import GuideSpec
from bayzflow.mcp.service import BayzflowService
from bayzflow.mcp.workloads import REGRESSION, regression_model


def _skill_root() -> Path:
    return Path(__file__).parent / "resources" / "skills" / "bayesian-inference"


def _skill_target(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    import os

    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills" / "bayesian-inference"
    return Path.home() / ".agents" / "skills" / "bayesian-inference"


def _json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bayzflow")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    guides = sub.add_parser("guide").add_subparsers(dest="guide_command", required=True)
    guides.add_parser("list")
    inspect_parser = guides.add_parser("inspect")
    inspect_parser.add_argument("kind")
    validate_parser = guides.add_parser("validate")
    validate_parser.add_argument("path")
    models = sub.add_parser("model").add_subparsers(dest="model_command", required=True)
    models.add_parser("inspect").add_argument("model_id", nargs="?", default="regression")
    skill = sub.add_parser("skill").add_subparsers(dest="skill_command", required=True)
    skill.add_parser("status")
    skill_install = skill.add_parser("install")
    skill_install.add_argument("--target")
    args = parser.parse_args(argv)

    service = BayzflowService()
    if args.command == "doctor":
        _json({**service.doctor(), "python": sys.version.split()[0], "mcp_extra": _module_available("mcp")})
    elif args.command == "model":
        service.register_model("regression", regression_model(), REGRESSION)
        _json(service.model_inspect(args.model_id))
    elif args.command == "guide":
        if args.guide_command == "list":
            _json(service.guide_list())
        elif args.guide_command == "inspect":
            _json(service.guide_schema(args.kind))
        else:
            path = Path(args.path)
            with path.open(encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
            if not isinstance(value, dict):
                raise ValueError("guide YAML must contain an object with kind and optional kwargs")
            spec = GuideSpec.from_dict(value)
            _json(service.guide_validate(spec.kind, dict(spec.kwargs)))
    elif args.command == "skill":
        root = _skill_root()
        if args.skill_command == "status":
            _json({"packaged": (root / "SKILL.md").is_file(), "path": str(root),
                   "installed": (_skill_target(None) / "SKILL.md").is_file(),
                   "installed_path": str(_skill_target(None))})
        else:
            target = _skill_target(args.target)
            if target.exists():
                raise FileExistsError(f"skill target already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(root, target)
            _json({"installed": True, "path": str(target)})
    return 0


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
