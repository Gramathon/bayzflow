"""Durable admission and bounded artifact reads for local CPU inference jobs."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import stat
import sys
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workloads import OUTPUT_FILES, run_job

PUBLIC_ID = re.compile(r"[A-Za-z0-9_.:-]{1,96}")
JOB_ID = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class Limits:
    max_samples: int = 256
    max_parameters: int = 100_000
    max_model_bytes: int = 8 * 1024**2
    max_seconds: int = 45
    max_artifact_bytes: int = 8 * 1024**2
    max_memory_bytes: int = 8 * 1024**3
    max_concurrent_jobs: int = 1
    max_epochs: int = 250
    max_jobs: int = 100

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("all resource limits must be positive integers")
        if self.max_samples > 1000:
            raise ValueError("max_samples cannot exceed the engine limit of 1000")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private_workspace(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if path.resolve() != path:
        raise ValueError("workspace must not traverse a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = path.stat()
    if not stat.S_ISDIR(st.st_mode) or st.st_mode & 0o077 or st.st_uid != os.getuid():
        raise ValueError("workspace must be a private directory owned by the operator")
    return path


class JobStore:
    def __init__(self, workspace: Path, limits: Limits):
        if sys.platform != "linux":
            raise ValueError("operational jobs require Linux resource limits")
        self.root = private_workspace(workspace)
        self.limits = limits
        self.db = self.root / "jobs.sqlite3"
        for suffix in ("", "-shm", "-wal", "-journal"):
            if Path(str(self.db) + suffix).is_symlink():
                raise ValueError("workspace database must not traverse a symlink")
        with closing(self.connect()) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, key TEXT UNIQUE, fingerprint TEXT, operation TEXT,
                    request TEXT, state TEXT, deadline REAL, error TEXT, artifacts TEXT
                )"""
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db)
        db.row_factory = sqlite3.Row
        return db

    def row(self, job_id: str) -> sqlite3.Row:
        if not JOB_ID.fullmatch(job_id):
            raise ValueError("invalid job id")
        with closing(self.connect()) as db, db:
            self.expire(db)
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("unknown job id")
        return row

    @staticmethod
    def expire(db: sqlite3.Connection) -> None:
        db.execute(
            "UPDATE jobs SET state='failed', error='deadline_exceeded' WHERE state='running' AND deadline <= ?",
            (time.time(),),
        )
        db.execute(
            "UPDATE jobs SET state='cancelled' WHERE state='cancelling' AND deadline <= ?",
            (time.time(),),
        )

    def cancel(self, job_id: str) -> dict:
        self.row(job_id)
        with closing(self.connect()) as db, db:
            db.execute("UPDATE jobs SET state='cancelling' WHERE id=? AND state='running'", (job_id,))
        return self.status(job_id)

    def status(self, job_id: str) -> dict:
        row = self.row(job_id)
        return {
            "job_id": row["id"],
            "state": row["state"],
            "operation": row["operation"],
            "error": row["error"],
            "artifacts": json.loads(row["artifacts"] or "[]"),
        }

    def submit(self, operation: str, request: dict, key: str, worker_args: tuple) -> dict:
        if not PUBLIC_ID.fullmatch(key):
            raise ValueError("invalid idempotency key")
        digest = fingerprint([operation, request])
        deadline = time.time() + self.limits.max_seconds
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.expire(db)
            existing = db.execute("SELECT id, fingerprint FROM jobs WHERE key=?", (key,)).fetchone()
            if existing:
                if existing["fingerprint"] != digest:
                    raise ValueError("idempotency key already used for a different request")
                job_id = existing["id"]
            else:
                if (
                    db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('running', 'cancelling')").fetchone()[0]
                    >= self.limits.max_concurrent_jobs
                ):
                    raise ValueError("job concurrency limit reached")
                if db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= self.limits.max_jobs:
                    raise ValueError("workspace job capacity reached")
                job_id = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,'running',?,NULL,NULL)",
                    (job_id, key, digest, operation, json.dumps(request), deadline),
                )
        if existing is None:
            threading.Thread(
                target=self._execute,
                daemon=True,
                args=(job_id, operation, request, deadline, worker_args),
            ).start()
        return self.status(job_id)

    def _execute(
        self, job_id: str, operation: str, request: dict, deadline: float, args: tuple
    ) -> None:
        directory = self.root / job_id
        state = "failed"
        error: str | None = "worker_failed"
        artifacts: list[dict] = []
        try:
            directory.mkdir(mode=0o700)
            process = multiprocessing.get_context("spawn").Process(
                target=run_job,
                args=(
                    directory,
                    operation,
                    request,
                    *args,
                    self.limits.max_memory_bytes,
                    self.limits.max_artifact_bytes,
                    deadline,
                ),
            )
            process.start()
            try:
                while process.is_alive():
                    remaining = deadline - time.time()
                    if self.row(job_id)["state"] in {"cancelling", "cancelled"}:
                        state, error = "cancelled", None
                    elif remaining <= 0:
                        error = "deadline_exceeded"
                    else:
                        process.join(min(remaining, 0.25))
                        continue
                    process.kill()
                    process.join()
                    break
                if state != "cancelled" and process.exitcode == 0 and time.time() < deadline:
                    artifacts = self._index(job_id, operation)
                    state, error = "succeeded", None
            finally:
                process.close()
        except (OSError, ValueError, TypeError, RuntimeError):
            error = "worker_or_artifact_rejected"
        with closing(self.connect()) as db, db:
            db.execute(
                """UPDATE jobs SET state=CASE WHEN state='cancelling' THEN 'cancelled' ELSE ? END,
                    error=CASE WHEN state='cancelling' THEN NULL ELSE ? END,
                    artifacts=CASE WHEN state='cancelling' THEN NULL ELSE ? END
                    WHERE id=? AND state IN ('running', 'cancelling')""",
                (state, error, json.dumps(artifacts), job_id),
            )

    @contextmanager
    def _file(self, job_id: str, name: str):
        dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(self.root, dir_flags)
        try:
            job_fd = os.open(job_id, dir_flags, dir_fd=root_fd)
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=job_fd)
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or info.st_size > self.limits.max_artifact_bytes
                    ):
                        raise ValueError("artifact rejected")
                    yield stream, info.st_size
            finally:
                os.close(job_fd)
        finally:
            os.close(root_fd)

    def _index(self, job_id: str, operation: str) -> list[dict]:
        artifacts = []
        total = 0
        for name in OUTPUT_FILES[operation]:
            with self._file(job_id, name) as (stream, size):
                total += size
                if total > self.limits.max_artifact_bytes:
                    raise ValueError("artifact budget exceeded")
                digest = hashlib.sha256(stream.read(size + 1)).hexdigest()
                artifacts.append(
                    {
                        "artifact_id": fingerprint([job_id, name, digest]),
                        "name": name,
                        "size": size,
                        "sha256": digest,
                    }
                )
        return artifacts

    def verified_checkpoint(self, job_id: str) -> Path:
        row = self.row(job_id)
        if row["state"] != "succeeded" or row["operation"] != "fit":
            raise ValueError("checkpoint must be a successful fit job")
        try:
            current = self._index(job_id, "fit")
        except OSError as exc:
            raise ValueError("checkpoint files rejected") from exc
        if json.loads(row["artifacts"]) != current:
            raise ValueError("checkpoint integrity check failed")
        return self.root / job_id

    def read(self, job_id: str, artifact_id: str, offset: int, length: int) -> dict:
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or not 1 <= length <= 32768
        ):
            raise ValueError("invalid artifact page")
        row = self.row(job_id)
        if row["state"] != "succeeded":
            raise ValueError("artifacts require a successful job")
        try:
            if self._index(job_id, row["operation"]) != json.loads(row["artifacts"]):
                raise ValueError("artifact integrity check failed")
            for artifact in json.loads(row["artifacts"]):
                if artifact["artifact_id"] == artifact_id:
                    with self._file(job_id, artifact["name"]) as (stream, size):
                        if offset > size:
                            raise ValueError("artifact offset exceeds size")
                        data = stream.read(size + 1)
                        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                            raise ValueError("artifact integrity check failed")
                        page = data[offset : offset + length]
                    return {
                        **artifact,
                        "offset": offset,
                        "next_offset": offset + len(page),
                        "eof": offset + len(page) == size,
                        "base64": base64.b64encode(page).decode("ascii"),
                    }
        except OSError as exc:
            raise ValueError("artifact files rejected") from exc
        raise ValueError("unknown artifact id")
