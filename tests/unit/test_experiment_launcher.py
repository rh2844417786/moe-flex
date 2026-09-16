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
