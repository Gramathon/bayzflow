import base64
import hashlib
import json
import os
import sys

import pytest
from torch import nn

from bayzflow.mcp.jobs import JobStore, Limits
from bayzflow.mcp.service import BayzflowService
from bayzflow.mcp.workloads import REGRESSION, regression_model


def test_workspace_and_admission(tmp_path):
    workspace = tmp_path / "workspace"
    service = BayzflowService(workspace=JobStore(workspace, Limits(max_parameters=1)))
    service.register_model("regression", regression_model(), REGRESSION)
    with pytest.raises(ValueError, match="model resource"):
        service.bayesian_plan("regression", ["0"], "auto_diag", {}, 1, 0, 1, 0.01)
    link = tmp_path / "link"
    link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        JobStore(link, Limits())


def test_fullrank_and_guide_configuration_are_checked_at_admission(tmp_path):
    service = BayzflowService(workspace=JobStore(tmp_path / "workspace", Limits()))
    service.register_model("wide", nn.Sequential(nn.Linear(512, 1)), REGRESSION)
    with pytest.raises(ValueError, match="at most 512 latent coordinates; found 513"):
        service.bayesian_plan("wide", ["0"], "auto_fullrank", {}, 1, 0, 1, 0.01)

    service.register_model("regression", regression_model(), REGRESSION)
    with pytest.raises(ValueError, match="create_plates must be null"):
        service.bayesian_plan(
            "regression", ["0"], "auto_normal", {"create_plates": "plain text"}, 1, 0, 1, 0.01
        )


def test_mcp_stdio_inference(tmp_path):
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession, StdioServerParameters, stdio_client

    workspace = tmp_path / "workspace"

    async def run():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "bayzflow.mcp.server", "--stdio", "--demo", "--workspace", str(workspace)],
            cwd=str(tmp_path),
            env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        )
        with (tmp_path / "mcp.stderr").open("w") as stderr:
            async with stdio_client(server, errlog=stderr) as (reader, writer), ClientSession(
                reader, writer,
            ) as client:
                await client.initialize()

                async def call(name, **kwargs):
                    reply = await client.call_tool(name, kwargs)
                    assert not reply.is_error, reply
                    return reply.structured_content or json.loads(reply.content[0].text)

                async def finished(job):
                    assert job["ok"], job
                    job_id = job["result"]["job_id"]
                    with anyio.fail_after(50):
                        while True:
                            status = (await call("job_status", job_id=job_id))["result"]
                            if status["state"] != "running":
                                assert status["state"] == "succeeded", status
                                return status
                            await anyio.sleep(0.1)

                plan_args = {
                    "model_id": "regression",
                    "layers": ["0"],
                    "kind": "auto_diag",
                    "kwargs": {},
                    "epochs": 3,
                }
                assert not (await call("bayesian_plan", **{**plan_args, "layers": ["../../x"]}))["ok"]
                plan = (await call("bayesian_plan", **plan_args))["result"]
                fit_args = {key: plan[key] for key in ("plan_id", "request")}
                assert not (
                    await call(
                        "posterior_fit", **{**fit_args, "plan_id": "tampered"}, idempotency_key="fit"
                    )
                )["ok"]
                job = await call("posterior_fit", **fit_args, idempotency_key="fit")
                assert not (await call("posterior_fit", **fit_args, idempotency_key="another-fit"))[
                    "ok"
                ]
                replay = await call("posterior_fit", **fit_args, idempotency_key="fit")
                assert replay["result"]["job_id"] == job["result"]["job_id"]
                checkpoint = (await finished(job))["job_id"]
                assert (await call("posterior_inspect", job_id=checkpoint))["result"]["guide"][
                    "kind"
                ] == "auto_diag"
                args = {"checkpoint_id": checkpoint, "samples": 5, "seed": 12}
                assert not (
                    await call(
                        "posterior_sample", **{**args, "samples": 257}, idempotency_key="oversize"
                    )
                )["ok"]
                for tool, selection in (
                    ("posterior_sample", {"sites": ["0.weight"]}),
                    ("measure_epistemic", {}),
                    ("measure_layers", {"layers": ["0"]}),
                ):
                    job_id = (
                        await finished(await call(tool, **args, **selection, idempotency_key=tool))
                    )["job_id"]
                    entries = (await call("artifact_list", job_id=job_id))["result"]
                    for entry in entries:
                        artifact_args = {"job_id": job_id, "artifact_id": entry["artifact_id"]}
                        page = (await call("artifact_read", **artifact_args))["result"]
                        raw = base64.b64decode(page["base64"])
                        assert page["eof"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
                        if entry["name"] == "measurement.json":
                            expected = (
                                "predictive_epistemic_variance"
                                if tool == "measure_epistemic"
                                else "parameter_variance"
                            )
                            assert json.loads(raw)["measurement"] == expected
                    assert not (
                        await call("artifact_read", job_id=job_id, artifact_id="../../etc/passwd")
                    )["ok"]
                target = workspace / checkpoint / "meta.json"
                target.unlink()
                target.symlink_to(tmp_path / "outside.json")
                assert not (await call("posterior_inspect", job_id=checkpoint))["ok"]
                return checkpoint

    checkpoint = anyio.run(run)
    assert JobStore(workspace, Limits()).status(checkpoint)["state"] == "succeeded"


def test_worker_deadline_and_cancellation(tmp_path):
    import time

    for expected in ("failed", "cancelled"):
        limits = Limits(max_seconds=1 if expected == "failed" else 30)
        service = BayzflowService(workspace=JobStore(tmp_path / expected, limits))
        service.register_model("regression", regression_model(), REGRESSION)
        plan = service.bayesian_plan("regression", ["0"], "auto_diag", {}, 250, 0, 1, .01)
        job_id = service.fit(plan["plan_id"], plan["request"], "fit")["job_id"]
        if expected == "cancelled":
            service.workspace.cancel(job_id)
        for _ in range(150):
            job = service.workspace.status(job_id)
            if job["state"] not in {"running", "cancelling"}:
                break
            time.sleep(.1)
        assert job["state"] == expected
        assert not job["artifacts"]
        if expected == "failed":
            assert job["error"] == "deadline_exceeded"
