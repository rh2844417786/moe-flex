"""Execute real shell builders and local recorders at the Docker boundary."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_experiment_autorun import ROOT, command, modules, valid


def test_owned_container_invocations_get_fresh_cidfiles_and_exact_labels(tmp_path):
    script = ROOT / "scripts/server/run_container.sh"
    cid_dir = tmp_path / "attempt/cids"
    cid_dir.mkdir(parents=True)
    code = """
source "$1"
project_root="$2"; git_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
FLEXMOE_CONTAINER_CID_DIR="$3"
FLEXMOE_CONTAINER_OWNER=owner123
FLEXMOE_CONTAINER_SUITE=suite
FLEXMOE_CONTAINER_SHA="$git_sha"
container_owned_arguments
printf '%s\n' "${owned_args[@]}"
container_owned_arguments
printf '%s\n' "${owned_args[@]}"
"""
    result = subprocess.run(
        ["bash", "-c", code, "test", str(script), str(tmp_path), str(cid_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    cids = [args[i + 1] for i, value in enumerate(args) if value == "--cidfile"]
    assert len(cids) == 2 and len(set(cids)) == 2
    assert all(Path(x).parent.is_dir() and not Path(x).exists() for x in cids)
    assert args.count("io.moe-flex.owner=owner123") == 2
    assert args.count("io.moe-flex.suite=suite") == 2
    assert args.count("io.moe-flex.sha=" + "a" * 40) == 2


def docker_recorder(tmp_path, *, owner="owner", inspect_failure=False):
    path = tmp_path / "docker"
    path.write_text(f"""#!{sys.executable}
import json,os,pathlib,sys
base=pathlib.Path(os.environ['RECORDER_ROOT'])
args=sys.argv[1:]
with (base/'calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
if args[0]=='inspect':
    if {inspect_failure!r}:
        print('daemon unavailable',file=sys.stderr); sys.exit(1)
    print(json.dumps([{{'Id': 'a'*64,'Config': {{'Labels': {{'io.moe-flex.owner': {owner!r},'io.moe-flex.suite':'suite','io.moe-flex.sha':'b'*40}}}}}}]))
elif args[0]=='ps': print('')
elif args[0]=='rm': pass
else: sys.exit(20)
""")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "owner,failure,want",
    [("owner", False, True), ("foreign", False, False), ("owner", True, False)],
)
def test_exact_cid_cleanup_never_removes_foreign_or_ambiguous(
    tmp_path, monkeypatch, owner, failure, want
):
    m = modules()
    state_module = sys.modules[m.PointExecutor.__module__]
    docker_recorder(tmp_path, owner=owner, inspect_failure=failure)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("RECORDER_ROOT", str(tmp_path))
    cid = tmp_path / "cids/one/container.cid"
    cid.parent.mkdir(parents=True)
    cid.write_text("a" * 64)
    result = state_module.cleanup_containers(
        tmp_path,
        {"cid_dir": str(tmp_path / "cids"), "owner": "owner"},
        "b" * 40,
        "suite",
    )
    assert result is want
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    removals = [row for row in calls if row[0] == "rm"]
    assert removals == ([["rm", "-f", "a" * 64]] if want else [])


def test_failed_preflight_stops_before_main_child(tmp_path):
    m = modules()
    with m.ExperimentState(tmp_path, "suite", {"sha": "a" * 40}) as state:
        executor = m.PointExecutor(state, timeout_s=1, grace_s=0.1)
        with pytest.raises(m.SafetyStop):
            executor.run(
                "gpu",
                command,
                valid,
                preflight=lambda p: [sys.executable, "-c", "import sys; sys.exit(2)"],
            )
        attempt = state.data["points"]["gpu"]["attempts"][0]
        assert not (Path(attempt["directory"]) / "result.json").exists()
        assert attempt["reason"] == "preflight-failed"


@pytest.mark.parametrize(
    "message,listed,ps_code,want",
    [
        ("error: no such object: {cid}\n", False, 0, True),
        ("Error: No such object: {cid}\n", False, 0, True),
        ("Error response from daemon: No such container: {cid}\n", False, 0, True),
        ("error: no such object: {cid}\n", True, 0, False),
        ("error: no such object: {cid}\n", False, 1, False),
        ("permission denied", False, 0, False),
        ("error: no such object: other-container\n", False, 0, False),
    ],
)
def test_auto_removed_cid_requires_matching_absence_and_live_inventory(
    tmp_path, monkeypatch, message, listed, ps_code, want
):
    m = modules()
    state_module = sys.modules[m.PointExecutor.__module__]
    cid = "a" * 64
    path = tmp_path / "cids/one/container.cid"
    path.parent.mkdir(parents=True)
    path.write_text(cid)

    def docker(*args):
        if args == ("inspect", cid):
            return subprocess.CompletedProcess(args, 1, "\n", message.format(cid=cid))
        if args == ("ps", "-aq", "--no-trunc"):
            return subprocess.CompletedProcess(args, ps_code, cid if listed else "", "")
        raise AssertionError(f"Absent or ambiguous containers must not be removed: {args}")

    monkeypatch.setattr(state_module, "_docker", docker)
    assert state_module.cleanup_containers(
        tmp_path, {"cid_dir": str(path.parent.parent), "owner": "owner"}, "b" * 40, "suite"
    ) is want


def test_successful_preflight_preserves_cleanup_failure_reason(tmp_path, monkeypatch):
    m = modules()
    state_module = sys.modules[m.PointExecutor.__module__]

    def docker(*args):
        assert args[0] in ("inspect", "ps")
        return subprocess.CompletedProcess(args, 1, "", "daemon unavailable")

    monkeypatch.setattr(state_module, "_docker", docker)
    code = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "c=p.parent/'cids'/'one'; c.mkdir(); "
        "(c/'container.cid').write_text('a'*64); p.write_text('{\"ok\": true}')"
    )
    with m.ExperimentState(tmp_path, "suite", {"sha": "b" * 40}) as state:
        executor = m.PointExecutor(state, timeout_s=2, grace_s=0.1)
        with pytest.raises(m.SafetyStop, match="cleanup-unproven"):
            executor.run(
                "corpus", command, valid,
                preflight=lambda p: [sys.executable, "-c", code, str(p)],
            )
        attempt = state.data["points"]["corpus"]["attempts"][0]
        assert attempt["reason"] == "cleanup-unproven"
        assert attempt["preflight_exit_code"] == 0
        assert attempt["cleanup"] == "unproven"
        assert not (Path(attempt["directory"]) / "result.json").exists()


def test_sigterm_persists_interrupted_and_resume_runs_fresh_attempt(tmp_path):
    m = modules()
    runner = tmp_path / "driver.py"
    # Execute the production lifecycle in another process to deliver a real signal.
    runner.write_text(f"""
import importlib.util, pathlib, signal, sys
spec=importlib.util.spec_from_file_location('autorun', {str(ROOT / "src/flexmoe/analysis/experiment_autorun.py")!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
signal.signal(signal.SIGTERM,m.interrupted)
root=pathlib.Path({str(tmp_path)!r})
with m.ExperimentState(root,'suite',{{'sha':'a'*40}}) as state:
    executor=m.PointExecutor(state,timeout_s=30,grace_s=.1)
    def factory(d,i):
        return {{'argv':[sys.executable,'-c','import time; time.sleep(30)'],'artifacts':[str(d/'absent.json')]}}
    try: executor.run('point',factory,lambda a: True)
    except InterruptedError: pass
""")
    process = subprocess.Popen([sys.executable, "-S", str(runner)])
    checkpoint = tmp_path / "runs/experiment-suite/suite/state.json"
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if checkpoint.exists():
                data = json.loads(checkpoint.read_text())
                attempt = (
                    data.get("points", {}).get("point", {}).get("attempts", [{}])[-1]
                )
                if attempt.get("command_pid"):
                    break
            time.sleep(0.01)
        else:
            pytest.fail("child never started")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    data = json.loads(checkpoint.read_text())
    old = data["points"]["point"]["attempts"][0]
    assert old["status"] == "interrupted" and old["cleanup"] == "complete"
    with pytest.raises(ProcessLookupError):
        os.kill(old["command_pid"], 0)
    with m.ExperimentState(tmp_path, "suite", {"sha": "a" * 40}, resume=True) as state:
        fresh = m.PointExecutor(state, timeout_s=1).run("point", command, valid)
        assert fresh["status"] == "complete" and fresh["number"] == 2


def test_launcher_actual_dry_run_and_no_implicit_gpu_selection(tmp_path):
    script = ROOT / "scripts/server/run_all_experiments.sh"
    assert script.exists(), "one-command launcher missing"
    env = {
        **os.environ,
        "GPU_IDS": "2,4,5,7",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"gpu_ids": "2,4,5,7"' in result.stdout
    assert '"executes_gpu": false' in result.stdout
    for value in ("", "0,0,2,3", "0,1,2"):
        failed = subprocess.run(
            ["bash", str(script), "--dry-run"],
            env={**env, "GPU_IDS": value},
            capture_output=True,
            text=True,
            check=False,
        )
        assert failed.returncode != 0


def test_reserved_cid_without_file_needs_reachable_daemon_proof(tmp_path, monkeypatch):
    m = modules()
    state_module = sys.modules[m.PointExecutor.__module__]
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 1\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    directory = tmp_path / "cids/reserved"
    directory.mkdir(parents=True)
    assert not state_module.cleanup_containers(
        tmp_path,
        {"cid_dir": str(directory.parent), "owner": "owner"},
        "a" * 40,
        "suite",
    )


@pytest.mark.parametrize("label", ["command", "preflight"])
@pytest.mark.parametrize("failure", ["sigint", "sigterm", "io"])
def test_first_pid_checkpoint_failure_reaps_child_before_cid_cleanup(
    tmp_path, monkeypatch, label, failure
):
    """A failed first PID save must not leave a launcher able to create a CID."""
    m = modules()
    state_module = sys.modules[m.PointExecutor.__module__]
    cleanup = state_module.cleanup_containers
    observed_alive = []
    child_pid = None

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def observe_cleanup(*args):
        observed_alive.append(alive(child_pid))
        return cleanup(*args)

    config = {"sha": "a" * 40}
    try:
        with m.ExperimentState(tmp_path, "suite", config) as state:
            save = state.save

            def fail_first_pid_save():
                nonlocal child_pid
                attempts = state.data["points"].get("point", {}).get("attempts", [])
                if attempts and child_pid is None and f"{label}_pid" in attempts[-1]:
                    child_pid = attempts[-1][f"{label}_pid"]
                    assert alive(child_pid)
                    if failure == "io":
                        raise OSError("injected first PID checkpoint write failure")
                    m.interrupted(
                        signal.SIGINT if failure == "sigint" else signal.SIGTERM, None
                    )
                save()

            monkeypatch.setattr(state, "save", fail_first_pid_save)
            monkeypatch.setattr(state_module, "cleanup_containers", observe_cleanup)
            executor = m.PointExecutor(state, timeout_s=30, grace_s=0.1)
            preflight = (
                (lambda p: [sys.executable, "-c", "import time; time.sleep(30)"])
                if label == "preflight"
                else None
            )

            def run():
                return executor.run(
                    "point",
                    lambda d, i: command(d, i, sleep=30),
                    valid,
                    preflight=preflight,
                )

            if failure != "io":
                with pytest.raises(InterruptedError):
                    run()
            elif label == "preflight":
                with pytest.raises(m.SafetyStop):
                    run()
            else:
                assert run()["status"] == "failed"
            saved = json.loads((state.directory / "state.json").read_text())
            old = saved["points"]["point"]["attempts"][0]
            assert old["status"] == ("failed" if failure == "io" else "interrupted")
            assert old["cleanup"] == "complete"
            assert child_pid is not None
            assert observed_alive == [False], "CID cleanup ran with a live launcher"
            assert not alive(child_pid), "owned child was not reaped"
        monkeypatch.setattr(state_module, "cleanup_containers", cleanup)
        with m.ExperimentState(tmp_path, "suite", config, resume=True) as state:
            result = m.PointExecutor(
                state, timeout_s=1, grace_s=0.1, retry_failed=failure == "io"
            ).run("point", command, valid)
            assert result["status"] == "complete" and result["number"] == 2
            assert Path(old["directory"]).is_dir()
    finally:
        # RED runs must not leak the deliberately exposed, test-owned process.
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child_pid, 0)
            except ChildProcessError:
                pass


def host_identity_recorder(tmp_path, monkeypatch):
    """Only host executables are doubled; CLI/config/state/export stay real."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gpu = binaries / "nvidia-smi"
    gpu.write_text(f"""#!{sys.executable}
import json,pathlib,sys
root=pathlib.Path({str(tmp_path)!r})
args=sys.argv[1:]
with (root/'inventory-calls.jsonl').open('a') as stream: stream.write(json.dumps(args)+'\\n')
assert args == ['--query-gpu=index,uuid','--format=csv,noheader','--id=2,4,5,7'], args
print((root/'inventory.txt').read_text(),end='')
""")
    gpu.chmod(0o755)
    git = binaries / "git"
    git.write_text(f"""#!{sys.executable}
import sys
args=sys.argv[1:]
if args == ['rev-parse','HEAD']: print('a'*40)
elif args == ['status','--porcelain','--','.',' :(exclude)docs/results/decode-mechanism-*'.strip()]: pass
else: raise AssertionError(args)
""")
    git.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("GPU_IDS", "2,4,5,7")
    rows = [
        f"{index}, GPU-{digit * 8}-{digit * 4}-{digit * 4}-{digit * 4}-{digit * 12}"
        for index, digit in ((2, "a"), (4, "b"), (5, "c"), (7, "d"))
    ]
    (tmp_path / "inventory.txt").write_text("\n".join(rows) + "\n")
    return rows


def identity_cli_fixture(tmp_path, monkeypatch):
    m = modules()
    rows = host_identity_recorder(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path)
    for relative in (m.decode_suite.DATA, m.decode_suite.MANIFEST):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")

    def bounded_pipeline(suite):
        marker = tmp_path / "pipeline-entry.log"
        with marker.open("a") as stream:
            stream.write("entered\n")
        suite.executor.run("cached-point", command, valid)

    monkeypatch.setattr(m.Suite, "pipeline", bounded_pipeline)
    args = ["run", "--run-id", "suite", "--project-root", str(tmp_path)]
    return m, rows, args


def test_cli_resume_binds_live_physical_identity_before_reuse_or_launch(
    tmp_path, monkeypatch
):
    import tarfile

    m, rows, args = identity_cli_fixture(tmp_path, monkeypatch)
    assert m.main(args) == 0
    checkpoint = tmp_path / "runs/experiment-suite/suite/state.json"
    original = json.loads(checkpoint.read_text())
    first = original["points"]["cached-point"]["attempts"][0]
    assert original["config"]["gpu_inventory"] == [
        {"index": int(row.split(",")[0]), "uuid": row.split(", ")[1]} for row in rows
    ]
    # CLI output order changes do not change index->physical UUID mapping.
    (tmp_path / "inventory.txt").write_text("\n".join(reversed(rows)) + "\n")
    assert m.main([*args, "--resume"]) == 0
    same = json.loads(checkpoint.read_text())
    assert len(same["points"]["cached-point"]["attempts"]) == 1
    assert (
        same["points"]["cached-point"]["attempts"][0]["directory"] == first["directory"]
    )
    with tarfile.open(same["public_archive"]) as archive:
        for member in archive.getmembers():
            contents = archive.extractfile(member).read().decode()
            assert "GPU-" not in contents
            assert "gpu_inventory" not in contents
    before = checkpoint.read_bytes()
    changed = [rows[0].replace("a", "e"), *rows[1:]]
    (tmp_path / "inventory.txt").write_text("\n".join(changed) + "\n")
    assert m.main([*args, "--resume"]) == 2
    assert checkpoint.read_bytes() == before
    assert (tmp_path / "pipeline-entry.log").read_text().splitlines() == [
        "entered",
        "entered",
    ]
    assert len((tmp_path / "inventory-calls.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate-index",
        "duplicate-uuid",
        "malformed",
        "unselected",
        "empty",
    ],
)
def test_cli_rejects_incomplete_or_ambiguous_host_gpu_identity(
    tmp_path, monkeypatch, defect
):
    m, rows, args = identity_cli_fixture(tmp_path, monkeypatch)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate-index":
        rows[1] = rows[1].replace("4,", "2,")
    elif defect == "duplicate-uuid":
        rows[1] = "4, " + rows[0].split(", ")[1]
    elif defect == "malformed":
        rows[0] = "2, GPU-not-a-uuid"
    elif defect == "unselected":
        rows[0] = rows[0].replace("2,", "3,")
    else:
        rows = []
    (tmp_path / "inventory.txt").write_text("\n".join(rows))
    assert m.main(args) == 2
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "pipeline-entry.log").exists()


def test_dry_run_and_status_do_not_probe_physical_gpu_inventory(tmp_path, monkeypatch):
    m, _, args = identity_cli_fixture(tmp_path, monkeypatch)
    assert m.main([*args, "--dry-run"]) == 0
    assert not (tmp_path / "inventory-calls.jsonl").exists()
    with m.ExperimentState(tmp_path, "suite", {}) as state:
        state.data["status"] = "partial"
        state.save()
    assert m.main(["status", "--run-id", "suite", "--project-root", str(tmp_path)]) == 0
    assert not (tmp_path / "inventory-calls.jsonl").exists()
