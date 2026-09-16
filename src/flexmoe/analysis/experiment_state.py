"""Private checkpoints and owned process/container lifecycle for experiment suites."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any


class SafetyStop(RuntimeError):
    """No further GPU work is safe in this invocation."""


def safe_name(name: str, limit: int = 32) -> str:
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
        or ".." in name
        or len(name) > limit
    ):
        raise ValueError("safe bounded run ID required")
    return name


def bounded(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    if path.resolve() != path or not path.is_relative_to(root) or path == root:
        raise ValueError("path must be canonical and project bounded")
    return path


def atomic_json(root: Path, path: Path, value: object) -> None:
    bounded(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentState:
    def __init__(
        self,
        root: Path,
        run_id: str,
        config: Mapping[str, Any],
        *,
        resume: bool = False,
    ):
        self.root = root.resolve()
        self.run_id = safe_name(run_id)
        self.directory = bounded(
            self.root, self.root / "runs/experiment-suite" / run_id
        )
        self.config = dict(config)
        self.resume = resume
        self.data: dict[str, Any] = {}
        self.lock: IO[str] | None = None

    def __enter__(self) -> ExperimentState:  # noqa: PYI034 -- Python 3.10 stdlib only
        base = self.directory.parent
        base.mkdir(parents=True, exist_ok=True)
        lock_path = bounded(self.root, base / ".lock")
        self.lock = lock_path.open("a+")
        try:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("experiment suite is locked") from error
            if self.resume:
                self.data = read_json(bounded(self.root, self.directory / "state.json"))
                if (
                    self.data.get("config") != self.config
                    or self.data.get("run_id") != self.run_id
                ):
                    raise ValueError("resume configuration/SHA/GPU mismatch")
            else:
                self.directory.mkdir(exist_ok=False)
                self.data = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "config": self.config,
                    "points": {},
                    "status": "running",
                    "ep_offload": "unsupported",
                }
                self.save()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.lock is not None:
            self.lock.close()

    def save(self) -> None:
        atomic_json(self.root, self.directory / "state.json", self.data)

    def skip(self, key: str, reason: str) -> None:
        safe_name(key, 80)
        point = self.data["points"].setdefault(key, {"attempts": []})
        point.update(status="skipped", reason=reason)
        self.save()


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], text=True, capture_output=True, timeout=30, check=False
    )


def cleanup_containers(
    root: Path, attempt: Mapping[str, Any], sha: str, suite: str
) -> bool:
    """Inspect exact recorded CIDs before removal; any ambiguity fails closed."""
    try:
        directory = bounded(root, Path(attempt["cid_dir"]))
        if not directory.is_dir():
            return False
        for subdir in sorted(directory.iterdir()):
            bounded(root, subdir)
            if not subdir.is_dir():
                return False
            cidfile = bounded(root, subdir / "container.cid")
            if not cidfile.exists():
                # Docker may have accepted create before the client wrote its
                # cidfile. Only a reachable daemon proving no matching owner
                # exists permits continuation; never kill a guessed container.
                listing = _docker(
                    "ps",
                    "-aq",
                    "--no-trunc",
                    "--filter",
                    f"label=io.moe-flex.owner={attempt['owner']}",
                    "--filter",
                    f"label=io.moe-flex.suite={suite}",
                    "--filter",
                    f"label=io.moe-flex.sha={sha}",
                )
                if listing.returncode or listing.stdout.strip():
                    return False
                continue
            cid = cidfile.read_text().strip()
            if not re.fullmatch(r"[0-9a-f]{64}", cid):
                return False
            inspected = _docker("inspect", cid)
            if inspected.returncode:
                # An arbitrary inspect error is not proof of container absence.
                listing = _docker("ps", "-aq", "--no-trunc")
                if listing.returncode or cid in listing.stdout.splitlines():
                    return False
                if "No such" not in inspected.stderr:
                    return False
                continue
            rows = json.loads(inspected.stdout)
            if not isinstance(rows, list) or len(rows) != 1:
                return False
            row = rows[0]
            labels = row.get("Config", {}).get("Labels", {})
            if row.get("Id") != cid or any(
                labels.get(k) != v
                for k, v in {
                    "io.moe-flex.owner": attempt["owner"],
                    "io.moe-flex.suite": suite,
                    "io.moe-flex.sha": sha,
                }.items()
            ):
                return False
            removed = _docker("rm", "-f", cid)
            listing = _docker("ps", "-aq", "--no-trunc")
            if (
                removed.returncode
                or listing.returncode
                or cid in listing.stdout.splitlines()
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        return False


class PointExecutor:
    def __init__(
        self,
        state: ExperimentState,
        *,
        timeout_s: float,
        grace_s: float = 60,
        retry_failed: bool = False,
    ):
        self.state, self.timeout_s, self.grace_s = state, timeout_s, grace_s
        self.retry_failed = retry_failed

    def recover(self) -> None:
        for point in self.state.data["points"].values():
            for attempt in point.get("attempts", []):
                if (
                    attempt.get("status") in {"running", "interrupted"}
                    or attempt.get("cleanup") != "complete"
                ):
                    if not cleanup_containers(
                        self.state.root,
                        attempt,
                        self.state.config.get("sha", ""),
                        self.state.run_id,
                    ):
                        raise SafetyStop("orphan-cleanup-unproven")
                    attempt["cleanup"] = "complete"
                    if attempt["status"] == "running":
                        attempt["status"] = "interrupted"
        self.state.save()

    def _reusable(
        self, attempt: Mapping[str, Any], validate: Callable[[dict[str, Any]], bool]
    ) -> bool:
        try:
            if (
                attempt.get("status") != "complete"
                or attempt.get("cleanup") != "complete"
            ):
                return False
            hashes = attempt.get("artifact_hashes", {})
            return (
                bool(hashes)
                and all(
                    file_digest(bounded(self.state.root, Path(path))) == digest
                    for path, digest in hashes.items()
                )
                and validate(dict(attempt))
            )
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _process(
        self, argv: Sequence[str], attempt: dict[str, Any], label: str, timeout: float
    ) -> int:
        directory = Path(attempt["directory"])
        env = {
            **os.environ,
            "GPU_IDS": self.state.config.get("gpu_ids", ""),
            "FLEXMOE_CONTAINER_CID_DIR": attempt["cid_dir"],
            "FLEXMOE_CONTAINER_OWNER": attempt["owner"],
            "FLEXMOE_CONTAINER_SUITE": self.state.run_id,
            "FLEXMOE_CONTAINER_SHA": self.state.config.get("sha", ""),
        }
        with (
            (directory / f"{label}.stdout.log").open("x") as out,
            (directory / f"{label}.stderr.log").open("x") as err,
        ):
            process = subprocess.Popen(
                list(argv),
                cwd=self.state.root,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                # Checkpointing can raise on signals or I/O just like waiting;
                # once spawned, the launcher must stay inside this cleanup scope.
                attempt[f"{label}_pid"] = process.pid
                self.state.save()
                return process.wait(timeout=timeout)
            finally:
                # Kill only the process group created by this invocation. This
                # also handles descendants left after a successful wrapper exit.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=min(self.grace_s, 5))
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=max(self.grace_s, 1))

    def run(
        self,
        key: str,
        factory: Callable[[Path, str], dict[str, Any]],
        validate: Callable[[dict[str, Any]], bool],
        *,
        preflight: Callable[[Path], Sequence[str]] | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_name(key, 80)
        point = self.state.data["points"].setdefault(key, {"attempts": []})
        attempts = point["attempts"]
        if attempts:
            last = attempts[-1]
            if last.get("inputs") == inputs and self._reusable(last, validate):
                point["status"] = "complete"
                return dict(last)
            if last["status"] == "failed" and not self.retry_failed:
                point["status"] = "failed"
                return dict(last)
            if last.get("cleanup") != "complete" or last["status"] in {
                "running",
                "interrupted",
            }:
                self.recover()
        number = max((x["number"] for x in attempts), default=0) + 1
        while number <= 999:
            directory = bounded(
                self.state.root,
                self.state.directory / "points" / key / f"attempt-{number:03d}",
            )
            if not directory.exists():
                break
            cids = bounded(self.state.root, directory / "cids")
            if cids.exists() and any(cids.iterdir()):
                raise SafetyStop("untracked-attempt")
            number += 1
        else:
            raise SafetyStop("attempt-limit")
        directory.mkdir(parents=True, exist_ok=False)
        cid_dir = directory / "cids"
        cid_dir.mkdir()
        child_id = f"{self.state.run_id}-a{number:03d}"
        spec = factory(directory, child_id)
        for path in spec["artifacts"]:
            bounded(self.state.root, Path(path))
        attempt = {
            **spec,
            "inputs": dict(inputs) if inputs is not None else None,
            "number": number,
            "directory": str(directory),
            "cid_dir": str(cid_dir),
            "owner": uuid.uuid4().hex,
            "status": "running",
            "cleanup": "pending",
        }
        attempts.append(attempt)
        point["status"] = "running"
        point.pop("reason", None)
        self.state.save()
        print(f"{key}: attempt {number} started", flush=True)
        interrupted: BaseException | None = None
        unsafe = False
        try:
            if preflight is not None:
                preflight_path = directory / "preflight.json"
                code = self._process(
                    preflight(preflight_path),
                    attempt,
                    "preflight",
                    min(self.timeout_s, 300) + self.grace_s,
                )
                if code != 0 or read_json(preflight_path).get("ok") is not True:
                    raise SafetyStop("preflight-failed")
                if not cleanup_containers(
                    self.state.root,
                    attempt,
                    self.state.config.get("sha", ""),
                    self.state.run_id,
                ):
                    raise SafetyStop("cleanup-unproven")
            code = self._process(
                spec["argv"], attempt, "command", self.timeout_s + self.grace_s
            )
            attempt["exit_code"] = code
            if code != 0:
                attempt.update(status="failed", reason="child-failed")
            elif not validate(attempt):
                attempt.update(status="failed", reason="invalid-evidence")
            else:
                attempt["artifact_hashes"] = {
                    path: file_digest(bounded(self.state.root, Path(path)))
                    for path in attempt["artifacts"]
                }
                attempt.update(status="complete", reason=None)
        except subprocess.TimeoutExpired:
            attempt.update(status="failed", reason="timeout")
            if (
                preflight is not None
                and not (directory / "command.stdout.log").exists()
            ):
                unsafe = True
        except SafetyStop:
            attempt.update(status="failed", reason="preflight-failed")
            unsafe = True
        except (KeyboardInterrupt, InterruptedError) as error:
            attempt.update(status="interrupted", reason="signal")
            interrupted = error
        except PermissionError:
            attempt.update(status="failed", reason="artifact-permissions")
            if (
                preflight is not None
                and not (directory / "command.stdout.log").exists()
            ):
                unsafe = True
        except (OSError, ValueError, KeyError, TypeError):
            attempt.update(status="failed", reason="invalid-evidence")
            if (
                preflight is not None
                and not (directory / "command.stdout.log").exists()
            ):
                unsafe = True
        finally:
            clean = cleanup_containers(
                self.state.root,
                attempt,
                self.state.config.get("sha", ""),
                self.state.run_id,
            )
            attempt["cleanup"] = "complete" if clean else "unproven"
            point["status"] = attempt["status"]
            self.state.save()
            print(
                f"{key}: {attempt['status']} ({attempt.get('reason') or 'validated'})",
                flush=True,
            )
        if not clean or unsafe:
            raise SafetyStop("cleanup-unproven" if not clean else "preflight-failed")
        if interrupted is not None:
            raise interrupted
        return attempt
