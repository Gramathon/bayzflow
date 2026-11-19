import json
import re
import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

from bayzflow.cli import main
from bayzflow.mcp.service import BayzflowService


def test_readme_quickstart_runs_verbatim(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[1] / "README.md"
    quickstart = readme.read_text(encoding="utf-8").split("## Quickstart", 1)[1]
    snippet = re.search(r"```python\n(.*?)```", quickstart, re.DOTALL)
    assert snippet is not None
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "quickstart.py"
    script.write_text(snippet[1], encoding="utf-8")
    scope = runpy.run_path(str(script))
    assert scope["manifest"]["random_seed"] == 17
    assert scope["reloaded"]["logits"].shape == (32, 4, 1)
    assert torch.isfinite(scope["reloaded"]["logits"]).all()


def test_cli_guides_doctor_and_skill_install(tmp_path, capsys):
    assert main(["guide", "list"]) == 0
    guides = json.loads(capsys.readouterr().out)
    assert {row["kind"] for row in guides} >= {"auto_normal", "auto_diag", "auto_lowrank", "auto_fullrank"}

    assert main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "0.2.0"

    destination = tmp_path / "skills" / "bayesian-inference"
    assert main(["skill", "install", "--target", str(destination)]) == 0
    assert (destination / "SKILL.md").is_file()
    assert (destination / "references" / "security.md").is_file()


def test_service_exposes_only_registered_models_and_validates_guides():
    service = BayzflowService()
    assert service.model_list() == []
    model = nn.Sequential(nn.Linear(3, 2))
    service.register_model("toy", model)
    assert service.model_inspect("toy")["parameters"] == 8
    assert service.layer_candidates("toy")[0]["name"] == "0"
    assert service.guide_validate("auto_lowrank", {"rank": 1})["valid"]
    with pytest.raises(ValueError):
        service.model_inspect("../../etc/passwd")


def test_mcp_protocol_in_memory():
    pytest.importorskip("mcp")
    from mcp import Client

    from bayzflow.mcp.server import create_server

    async def run():
        async with Client(create_server()) as client:
            result = await client.list_tools()
            names = {tool.name for tool in result.tools}
            assert {"bayzflow_doctor", "guide_list", "guide_validate", "model_list"} <= names
            response = await client.call_tool("guide_validate", {"kind": "auto_lowrank", "kwargs": {"rank": 2}})
            assert not response.is_error
            invalid = await client.call_tool(
                "guide_validate",
                {"kind": "auto_normal", "kwargs": {"create_plates": "plain text"}},
            )
            assert invalid.is_error

    import anyio

    anyio.run(run)
