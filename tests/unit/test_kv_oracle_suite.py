import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_kv_oracle import fixture


def suite():
    import flexmoe.bench.kv_oracle_suite as module

    return module


def test_stdlib_cli_help():
    spec = importlib.util.find_spec("flexmoe.bench.kv_oracle_suite")
    assert spec is not None
    result = subprocess.run(
        [sys.executable, "-S", spec.origin, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(mode in result.stdout for mode in ("screen", "confirm", "export"))


def test_command_preserves_custom_paths_and_numeric_options(tmp_path):
    options = {
        "model_path": "/mounted/custom model",
        "dataset_path": "/project/custom data",
        "dataset_manifest": "/project/custom manifest",
        "batch_size": 32,
        "context_length": 4096,
        "output_length": 64,
        "repetitions": 3,
        "warmups": 1,
        "max_num_seqs": 512,
        "max_num_batched_tokens": 8192,
        "seed": 42,
        "small_add_bytes": 200,
        "large_add_bytes": 400,
        "safety_fraction": 0.8,
        "safety_margin_bytes": 100,
        "timeout_s": 123,
    }
    command = suite().point_command(
        tmp_path, "small", tmp_path / "small", tmp_path / "anchor", options
    )
    for name, value in options.items():
        assert command[command.index("--" + name.replace("_", "-")) + 1] == str(value)
    assert command[command.index("--baseline-reference") + 1] == str(
        tmp_path / "anchor"
    )


@pytest.mark.parametrize(
    "order,expected",
    [("forward", ["r0", "small", "large"]), ("reverse", ["large", "small", "r0"])],
)
def test_order_fresh_r0_and_anchor_capacity_only(
    tmp_path, monkeypatch, order, expected
):
    module = suite()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    row = fixture(reps=1)
    row["contract"].update(
        batch_size=512,
        output_length=256,
        source_request_count=512,
        unique_selected_request_count=512,
    )
    row["repetitions"][0].update(
        request_count=512, generated_tokens=131072, output_tokens_per_second=1310.72
    )
    (anchor / "summary.json").write_text(json.dumps(row))
    args = module.parser().parse_args(
        [
            "confirm",
            "--project-root",
            str(tmp_path),
            "--suite-id",
            "test-" + order,
            "--order",
            order,
            "--anchor",
            str(anchor),
        ]
    )
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="b" * 40)
        role = command[command.index("--role") + 1]
        calls.append(role)
        directory = Path(command[command.index("--run-dir") + 1])
        directory.mkdir()
        data = fixture(role, {"r0": 100, "small": 80, "large": 70}[role])
        data["contract"] = dict(row["contract"], repetitions_requested=3)
        data["run_id"] = directory.name
        for rep in data["repetitions"]:
            rep.update(
                request_count=512,
                generated_tokens=131072,
                output_tokens_per_second=131072 / rep["elapsed_s"],
            )
        (directory / "summary.json").write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    path = module.execute_suite(args)
    assert calls == expected
    export = json.loads(
        (
            tmp_path / "docs/results" / ("kv-oracle-test-" + order) / "results.json"
        ).read_text()
    )
    assert export["comparison"]["r0_elapsed_median_s"] == 100
    assert export["comparison"]["formal_offload_gain"] is False
    if order == "reverse":
        assert export["anchor"]["repetitions_completed"] == 1
    assert (path / "r0/summary.json").is_file()
    with pytest.raises(FileExistsError):
        module.execute_suite(args)


def test_failed_point_preserves_smoke_and_exports_missing_roles(tmp_path, monkeypatch):
    module = suite()
    args = module.parser().parse_args(
        ["screen", "--project-root", str(tmp_path), "--suite-id", "failed"]
    )

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="b" * 40)
        directory = Path(command[command.index("--run-dir") + 1])
        directory.mkdir()
        (directory / "smoke.json").write_text(json.dumps(fixture()["smoke"]))
        return SimpleNamespace(returncode=124)

    monkeypatch.setattr(module.subprocess, "run", run)
    module.execute_suite(args)
    data = json.loads(
        (tmp_path / "docs/results/kv-oracle-failed/results.json").read_text()
    )
    assert data["runs"][0]["status"] == "timeout"
    assert data["runs"][0]["smoke"]["generated_tokens"] == 8
    assert data["comparison"]["decision"] == "insufficient-evidence"
    assert data["runs"][1]["status"] == "failed"


def test_reverse_anchor_rejects_other_output_length_before_launch(
    tmp_path, monkeypatch
):
    module = suite()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    (anchor / "summary.json").write_text(json.dumps(fixture(reps=1)))
    args = module.parser().parse_args(
        [
            "confirm",
            "--project-root",
            str(tmp_path),
            "--order",
            "reverse",
            "--anchor",
            str(anchor),
        ]
    )
    with pytest.raises(ValueError):
        module.execute_suite(args)


def test_launcher_clean_guard_allows_only_generated_oracle_results(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    script = Path.cwd() / "scripts/server/run_kv_oracle.sh"

    def clean():
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; oracle_clean_code "$2"',
                "test",
                str(script),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    assert clean() == 0
    output = tmp_path / "docs/results/kv-oracle-screen/results.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    assert clean() == 0
    source = tmp_path / "runner.py"
    source.write_text("changed code")
    assert clean() != 0


@pytest.mark.parametrize("malformed", [False, True])
def test_direct_export_recovers_independent_smoke_without_readable_summary(
    tmp_path, malformed
):
    module = suite()
    source = tmp_path / "source"
    run = source / "r0"
    run.mkdir(parents=True)
    (source / "suite.json").write_text(
        json.dumps(
            {"roles": {r: r for r in ("r0", "small", "large")}, "order": "forward"}
        )
    )
    (run / "smoke.json").write_text(json.dumps(fixture()["smoke"]))
    if malformed:
        (run / "summary.json").write_text("{")
    result = module.export_suite(source, tmp_path / "export")
    assert result["runs"][0]["smoke"]["generated_tokens"] == 8
    assert result["runs"][0]["status"] == "failed"
    assert result["comparison"]["decision"] == "insufficient-evidence"
    if malformed:
        assert (run / "summary.json").read_text() == "{"


def test_malformed_launcher_summary_reaches_export_and_preserves_source(
    tmp_path, monkeypatch
):
    module = suite()
    args = module.parser().parse_args(
        ["screen", "--project-root", str(tmp_path), "--suite-id", "malformed"]
    )

    def run(command, **kwargs):
        directory = Path(command[command.index("--run-dir") + 1])
        directory.mkdir()
        (directory / "summary.json").write_text("{")
        (directory / "smoke.json").write_text(json.dumps(fixture()["smoke"]))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", run)
    source = module.execute_suite(args)
    assert (source / "r0/summary.json").read_text() == "{"
    result = json.loads(
        (tmp_path / "docs/results/kv-oracle-malformed/results.json").read_text()
    )
    assert result["runs"][0]["status"] == "failed"
    assert result["runs"][0]["smoke"]["generated_tokens"] == 8
    assert result["comparison"]["decision"] == "insufficient-evidence"
