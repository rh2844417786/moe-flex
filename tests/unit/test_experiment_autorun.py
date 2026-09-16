"""Real checkpoint/process tests with only GPU/Docker boundaries substituted."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_decode_suite import evidence

ROOT = Path(__file__).resolve().parents[2]


def modules():
    path = ROOT / "src/flexmoe/analysis/experiment_autorun.py"
    assert path.is_file(), "stdlib experiment orchestrator is missing"
    return importlib.import_module("flexmoe.analysis.experiment_autorun")


def command(directory, child_id, *, sleep=0, fail=False):
    code = (
        "import json,pathlib,sys,time; "
        "p=pathlib.Path(sys.argv[1]); "
        "p.write_text(json.dumps({'status':'complete'})); "
        f"time.sleep({sleep}); sys.exit({int(fail)})"
    )
    return {
        "argv": [sys.executable, "-c", code, str(directory / "result.json")],
        "artifacts": [str(directory / "result.json")],
        "child_id": child_id,
    }


def valid(spec):
    return json.loads(Path(spec["artifacts"][0]).read_text())["status"] == "complete"


def test_actual_executor_resume_preserves_success_and_interrupted_attempt(tmp_path):
    m = modules()
    config = {"sha": "a" * 40, "gpu_ids": "0,1,2,3", "timeout_s": 1}
    with m.ExperimentState(tmp_path, "suite", config) as state:
        executor = m.PointExecutor(state, timeout_s=0.1, grace_s=0.1)
        a = executor.run("point-a", command, valid)
        assert a["status"] == "complete"
        b = executor.run("point-b", lambda d, i: command(d, i, sleep=2), valid)
        assert b["reason"] == "timeout"
        # Emulate power loss after the durable running write, before completion.
        b["status"] = "running"
        state.save()
        old = Path(b["directory"]) / "result.json"
        assert old.is_file()
    with m.ExperimentState(tmp_path, "suite", config, resume=True) as state:
        executor = m.PointExecutor(state, timeout_s=1, grace_s=0.1)
        assert executor.run("point-a", command, valid)["directory"] == a["directory"]
        fresh = executor.run("point-b", command, valid)
        assert fresh["status"] == "complete"
        assert fresh["directory"] != b["directory"]
        assert old.is_file()
        assert len(state.data["points"]["point-a"]["attempts"]) == 1
        assert len(state.data["points"]["point-b"]["attempts"]) == 2
    with (
        pytest.raises(ValueError, match="configuration"),
        m.ExperimentState(tmp_path, "suite", {**config, "sha": "b" * 40}, resume=True),
    ):
        pass


def test_lock_failed_retry_and_corrupt_success(tmp_path):
    m = modules()
    config = {"sha": "a" * 40, "gpu_ids": "0,1,2,3"}
    with m.ExperimentState(tmp_path, "s", config) as state:
        with (
            pytest.raises(RuntimeError, match="locked"),
            m.ExperimentState(tmp_path, "other", config),
        ):
            pass
        ex = m.PointExecutor(state, timeout_s=1, grace_s=0.1)
        ex.run("failure", lambda d, i: command(d, i, fail=True), valid)
        good = ex.run("good", command, valid)
        Path(good["artifacts"][0]).write_text("corrupt")
    with m.ExperimentState(tmp_path, "s", config, resume=True) as state:
        ex = m.PointExecutor(state, timeout_s=1, grace_s=0.1)
        assert ex.run("failure", command, valid)["status"] == "failed"
        assert ex.run("good", command, valid)["status"] == "complete"
        assert len(state.data["points"]["failure"]["attempts"]) == 1
        assert len(state.data["points"]["good"]["attempts"]) == 2
    with m.ExperimentState(tmp_path, "s", config, resume=True) as state:
        ex = m.PointExecutor(state, timeout_s=1, grace_s=0.1, retry_failed=True)
        assert ex.run("failure", command, valid)["status"] == "complete"
        assert len(state.data["points"]["failure"]["attempts"]) == 2


@pytest.mark.parametrize("name", ["../escape", "x/child", "a..b", ".hidden"])
def test_run_ids_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError), modules().ExperimentState(tmp_path, name, {}):
        pass


def test_symlink_output_is_rejected(tmp_path):
    (tmp_path / "runs").symlink_to(tmp_path.parent, target_is_directory=True)
    with (
        pytest.raises(ValueError, match="canonical"),
        modules().ExperimentState(tmp_path, "suite", {}),
    ):
        pass


def native_memory():
    raw = evidence(mode="native", kv=1000)
    raw["contract"]["physical_safety_reserve_bytes"] = 2_000_000_000
    for group in [
        raw["memory"],
        raw["final_memory"],
        *[r["memory"] for r in raw["repetitions"]],
    ]:
        for row in group:
            row["total_gpu_bytes"] = 80_000_000_000
            row["free_gpu_bytes"] = 10_000_000_000
    return raw


def test_native_memory_drives_aligned_kv_and_rejects_missing_or_wrong_sha():
    m = modules()
    raw = native_memory()
    assert m.select_kv(raw, "a" * 40) == (500, 1000)
    for broken in ("missing", "rank", "reserve", "block"):
        bad = copy.deepcopy(raw)
        if broken == "missing":
            del bad["memory"][0]["torch_peak_reserved_bytes"]
        elif broken == "rank":
            bad["memory"][3]["rank"] = 0
        elif broken == "reserve":
            bad["final_memory"][1]["free_gpu_bytes"] = 1
        else:
            bad["actual_kv"]["num_gpu_blocks"] = 3
        assert m.select_kv(bad, "a" * 40) is None
    assert m.select_kv(raw, "b" * 40) is None


def test_instrumented_coverage_is_reached_without_timing_eligibility():
    m = modules()
    raw = native_memory()
    prof = evidence(profile=True)
    prof["memory"], prof["final_memory"] = raw["memory"], raw["final_memory"]
    prof["contract"]["physical_safety_reserve_bytes"] = 2_000_000_000
    prof["contract"]["target_batch"] = 1000
    for rep in prof["repetitions"]:
        rep["memory"] = copy.deepcopy(raw["memory"])
        for obs in rep["worker_observations"]:
            obs["target_batch"] = 1000
    assert m.reached_summary(prof, "a" * 40, 1024, 1000)
    prof["repetitions"][2]["worker_observations"][3]["captured_steps"] = 63
    assert not m.reached_summary(prof, "a" * 40, 1024, 1000)
    assert m.representatives([1, 16, 50, 100, 200, 500, 1000]) == [1, 100, 1000]


def test_host_cli_bootstraps_without_site_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "src/flexmoe/analysis/experiment_autorun.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "status" in result.stdout


def test_archive_copies_only_allowlisted_regular_files(tmp_path):
    m = modules()
    public = tmp_path / "public"
    public.mkdir()
    (public / "summary.json").write_text('{"status":"complete"}')
    (public / "secret.log").write_text("private token IDs")
    archive = tmp_path / "report.tar.gz"
    m.archive_public(tmp_path, archive, [public / "summary.json"])
    import tarfile

    with tarfile.open(archive) as stream:
        assert stream.getnames() == ["public/summary.json"]
    (public / "evil.json").symlink_to(public / "secret.log")
    with pytest.raises(ValueError):
        m.archive_public(tmp_path, tmp_path / "bad.tar.gz", [public / "evil.json"])


def test_calibration_consumer_accepts_real_split_workload_abi(tmp_path):
    from test_decode_corpus import write_dataset

    from flexmoe.bench.expert_cache_runner import split_workload
    from flexmoe.datasets.decode_corpus import load_unique_workload

    m = modules()
    data, manifest = write_dataset(tmp_path, [(i, 2, 3, 4) for i in range(40)])
    prompts, _, metadata = load_unique_workload(
        data, manifest, context_length=4, request_count=8, calibration_count=32
    )
    metadata["evaluation_input_sha256"] = m.digest(prompts)
    actual = split_workload(data, manifest, 4, 8, 32)
    receipt = tmp_path / "corpus.json"
    receipt.write_text(json.dumps({"4": metadata}))
    with m.ExperimentState(tmp_path, "suite", {"sha": "a" * 40}) as state:
        state.data["points"]["corpus"] = {"attempts": [{"artifacts": [str(receipt)]}]}
        suite = m.Suite(state, m.PointExecutor(state, timeout_s=1))
        summary = {
            "commit": "a" * 40,
            "workload": actual.evaluation.metadata,
            "calibration_input_hashes": list(actual.hashes),
        }
        assert suite.verify_identity(summary, 4, 8, calibration=True)
        summary["workload"]["input_sha256"] = "wrong-context"
        assert not suite.verify_identity(summary, 4, 8, calibration=True)


def test_dependency_change_cannot_reuse_completed_step(tmp_path):
    m = modules()
    with m.ExperimentState(tmp_path, "suite", {}) as state:
        ex = m.PointExecutor(state, timeout_s=1)
        first = ex.run("a", command, valid, inputs={"profile": "old"})
        second = ex.run("a", command, valid, inputs={"profile": "new"})
        assert first["directory"] != second["directory"]
        assert second["number"] == 2


def test_decisions_freeze_after_first_materialization(tmp_path):
    m = modules()
    with m.ExperimentState(tmp_path, "suite", {"sha": "a" * 40}) as state:
        suite = m.Suite(state, m.PointExecutor(state, timeout_s=1))
        assert suite.decide("representatives", [1, 50, 100]) == [1, 50, 100]
        assert suite.decide("representatives", [1, 500, 1000]) == [1, 50, 100]
        assert suite.decide("kv", None) is None
        assert suite.decide("kv", [500, 1000]) == [500, 1000]
        assert suite.decide("kv", [600, 1200]) == [500, 1000]


def test_abandoned_attempt_directory_is_retained_on_resume(tmp_path):
    m = modules()
    with m.ExperimentState(tmp_path, "suite", {}) as state:
        abandoned = state.directory / "points/a/attempt-001"
        abandoned.mkdir(parents=True)
        (abandoned / "private.log").write_text("preserve me")
        result = m.PointExecutor(state, timeout_s=1).run("a", command, valid)
        assert result["number"] == 2
        assert (abandoned / "private.log").read_text() == "preserve me"


def test_real_plan_commands_keep_successful_calibration_path_and_fresh_leaf(tmp_path):
    m = modules()
    recorder = tmp_path / "scripts/server/run_decode_mechanism.sh"
    recorder.parent.mkdir(parents=True)
    recorder.write_text("""#!/usr/bin/env bash
exec "$RECORDER_PYTHON" -c 'import json,pathlib,sys; pathlib.Path("recorded.json").write_text(json.dumps(sys.argv[1:])); sys.exit(1)' "$@"
""")
    profile = tmp_path / "runs/decode-mechanism/suite-a003-cal-1024/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}")
    import os

    old = os.environ.get("RECORDER_PYTHON")
    os.environ["RECORDER_PYTHON"] = sys.executable
    try:
        with m.ExperimentState(
            tmp_path,
            "suite",
            {
                "sha": "a" * 40,
                "model_path": "a model",
                "dataset_path": "a data",
                "dataset_manifest": "manifest",
                "timeout_s": 1,
            },
        ) as state:
            state.data["points"]["cal-1024"] = {
                "status": "complete",
                "attempts": [
                    {
                        "source": str(profile.parent),
                        "status": "complete",
                        "cleanup": "complete",
                    }
                ],
            }
            suite = m.Suite(state, m.PointExecutor(state, timeout_s=1))
            suite.preflight = lambda p: [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"ok\":true}')",
                str(p),
            ]
            suite.decode("coverage-1024-50", "coverage", batches=[50])
            argv = json.loads((tmp_path / "recorded.json").read_text())
            assert argv[argv.index("--run-id") + 1] == "suite-a001-coverage-1024-50"
            assert argv[argv.index("--profile-path") + 1] == str(profile)
            assert argv[argv.index("--model-path") + 1] == "a model"
            assert argv[argv.index("--target-batch") + 1] == "50"
            assert argv[argv.index("--timeout-s") + 1] == "1"
    finally:
        if old is None:
            del os.environ["RECORDER_PYTHON"]
        else:
            os.environ["RECORDER_PYTHON"] = old


def test_partial_public_report_preserves_missing_parallel_rows_without_secrets(
    tmp_path,
):
    m = modules()
    with m.ExperimentState(tmp_path, "suite", {"sha": "a" * 40}) as state:
        state.skip("cal-1024", "calibration-unavailable")
        state.data["status"] = "partial"
        suite = m.Suite(state, m.PointExecutor(state, timeout_s=1))
        archive = suite.export()
        report = Path(state.data["public_report"])
        result = json.loads((report / "summary.json").read_text())
        assert result["ep_offload"] == "unsupported"
        rows = result["parallel_comparison"]["rows"]
        assert [row["config"] for row in rows] == ["tp4", "ep4-dp4", "ep4-dp2"]
        assert all(row["median_tokens_s"] is None for row in rows)
        assert result["tp4_abc"]["status"] == "unavailable"
        assert archive.is_file()
        assert str(tmp_path) not in (report / "summary.json").read_text()
        second = suite.export()
        assert second != archive and archive.is_file()


def test_finite_pipeline_continues_independent_failures_and_skips_unsafe_kv(
    tmp_path, monkeypatch
):
    m = modules()
    scripts = tmp_path / "scripts/server"
    scripts.mkdir(parents=True)
    config = {
        "sha": "a" * 40,
        "gpu_ids": "0,1,2,3",
        "model_path": "model",
        "dataset_path": "data",
        "dataset_manifest": "manifest",
        "timeout_s": 2,
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
    }
    selected = [m.digest(i) for i in range(1024)]
    corpus = {
        str(context): {
            "no_repetition": True,
            "dataset_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "calibration_input_hashes": [m.digest([context, i]) for i in range(32)],
            "selected_input_hashes": selected,
            "evaluation_input_sha256": m.digest(context),
        }
        for context in (1024, 4096)
    }
    (tmp_path / "fixture.json").write_text(json.dumps(corpus))
    recorder = tmp_path / "boundary.py"
    recorder.write_text("""import hashlib,json,pathlib,sys
root=pathlib.Path.cwd(); args=sys.argv[1:]
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
with (root/'events.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
def option(k): return args[args.index(k)+1]
if args[0]=='build':
    (root/'build').mkdir(exist_ok=True); (root/'build/image.env').write_text('IMAGE=fixture\\nGIT_SHA='+'a'*40+'\\n')
elif args[0]=='image': print('a'*40)
elif 'flexmoe.runtime.preflight' in args:
    pathlib.Path(option('--output')).write_text('{"ok":true}')
elif '-c' in args:
    pathlib.Path(args[-1]).write_text((root/'fixture.json').read_text())
elif 'flexmoe.bench.parallel_runner' in args:
    source=pathlib.Path(option('--run-dir')); assert not source.exists(); source.mkdir()
    (source/'summary.json').write_text(json.dumps({'config':option('--config'),'status':'failed'})); sys.exit(1)
else:
    source=root/'runs/decode-mechanism'/option('--run-id'); source.mkdir(parents=True)
    if args[0]!='calibrate':
        (source/'summary.json').write_text('{"status":"failed"}'); (source/'launcher.json').write_text('{"status":"failed","exit_code":1}'); sys.exit(1)
    context=option('--context-length'); corpus=json.loads((root/'fixture.json').read_text())[context]
    identity={'geometry':{'total_layers':2,'num_experts':4},'model_config_sha256':'a'*64,'model_identity_sha256':'b'*64,'tensor_parallel_size':4}
    profile=dict(identity,schema_version=1,calibration_input_hashes=corpus['calibration_input_hashes'],counts=[[1,2,3,4],[4,3,2,1]],forward_counts=[10,10])
    workload={k:corpus[k] for k in ('dataset_sha256','dataset_manifest_sha256')}
    workload.update(input_sha256=corpus['evaluation_input_sha256'],unique_selected_request_count=1024,repeated_request_count=0)
    summary=dict(status='complete',storage_backend='native-calibration',identity=identity,profile_sha256=digest(profile),commit='a'*40,workload=workload,calibration_input_hashes=corpus['calibration_input_hashes'])
    (source/'summary.json').write_text(json.dumps(summary)); (source/'profile.json').write_text(json.dumps(profile)); (source/'launcher.json').write_text('{"status":"complete","exit_code":0}')
""")
    for name, prefix in (
        ("run_container.sh", ""),
        ("run_decode_mechanism.sh", ""),
        ("build.sh", "build"),
    ):
        (scripts / name).write_text(
            f'#!/usr/bin/env bash\nexec "{sys.executable}" "{recorder}" {prefix} "$@"\n'
        )
    docker = tmp_path / "docker"
    docker.write_text(
        f'#!/usr/bin/env bash\nexec "{sys.executable}" "{recorder}" "$@"\n'
    )
    docker.chmod(0o755)
    import os

    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with m.ExperimentState(tmp_path, "suite", config) as state:
        suite = m.Suite(state, m.PointExecutor(state, timeout_s=2))
        suite.pipeline()
        points = state.data["points"]
        assert (
            points["cal-1024"]["status"] == points["cal-4096"]["status"] == "complete"
        )
        assert all(
            points[f"parallel-{name}"]["status"] == "failed"
            for name in ("tp4", "ep4-dp4", "ep4-dp2")
        )
        assert len([key for key in points if key.startswith("coverage-1024-")]) == 7
        assert points["kv"]["reason"] == "native-memory-unavailable"
        assert points["coverage-4096"]["reason"] == "no-reached-batch"
        assert points["offload"]["status"] == points["overhead"]["status"] == "skipped"
        state.data["status"] = "partial"
        suite.export()
        events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        parallel = [
            event for event in events if "flexmoe.bench.parallel_runner" in event
        ]
        assert len(parallel) == 3
        decoded = [
            event
            for event in events
            if event[0] in ("calibrate", "native", "matched-resident", "offload")
        ]
        assert len(decoded) == 10
        assert all(
            "suite-a001-" in event[event.index("--run-id") + 1] for event in decoded
        )


def test_positive_pipeline_consumes_actual_plan_calibration_and_observed_kv(tmp_path):
    """Plan boundary test; native/reached evidence is checked, no GPU claims."""
    m = modules()
    commands = []
    native = native_memory()
    assert m.decode_suite.validate_run(native)["timing_eligible"]
    coverage = evidence(profile=True)
    coverage["contract"]["target_batch"] = 1000
    assert m.reached_summary(coverage, "a" * 40, 1024, 1000)

    class EvidenceExecutor:
        def recover(self):
            pass

        def run(self, key, factory, validate, **kwargs):
            directory = state.directory / key
            directory.mkdir()
            # Simulate a successful resumed calibration in attempt 3.
            spec = factory(
                directory, "suite-a003" if key.startswith("cal-") else "suite-a001"
            )
            commands.append((key, spec))
            if key.startswith("cal-"):
                source = Path(spec["source"])
                source.mkdir(parents=True)
                (source / "profile.json").write_text('{"calibration":3}')
            if key == "native":
                source = Path(spec["source"])
                source.mkdir(parents=True)
                (source / "summary.json").write_text(json.dumps(native))
            # Coverage status supplied at the process boundary is based on
            # the independently validated instrumented fixture above.
            state.data["points"][key] = {
                "status": "complete",
                "attempts": [dict(spec, status="complete")],
            }

    config = {
        "sha": "a" * 40,
        "model_path": "model",
        "dataset_path": "data",
        "dataset_manifest": "manifest",
        "timeout_s": 7200,
    }
    with m.ExperimentState(tmp_path, "suite", config) as state:
        suite = m.Suite(state, EvidenceExecutor())
        suite.pipeline()
        by_key = dict(commands)
        assert state.data["decisions"]["representatives"] == [1, 100, 1000]
        assert state.data["decisions"]["kv"] == [500, 1000]
        assert len([key for key in by_key if key.startswith("coverage-4096-")]) == 3
        assert len([key for key in by_key if key.startswith("offload-")]) == 9
        for key, spec in commands:
            if spec.get("kind") != "decode":
                continue
            argv = spec["argv"]
            context = argv[argv.index("--context-length") + 1]
            assert argv[argv.index("--profile-path") + 1].endswith(
                f"suite-a003-cal-{context}/profile.json"
            )
        for key, amount in (
            ("kv-a", "500"),
            ("kv-b", "500"),
            ("kv-c", "1000"),
            ("overhead-0", "500"),
            ("overhead-1", "500"),
        ):
            argv = by_key[key]["argv"]
            assert argv[argv.index("--kv-bytes") + 1] == amount
        assert "--profile" not in by_key["overhead-0"]["argv"]
        assert "--profile" in by_key["overhead-1"]["argv"]
