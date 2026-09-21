import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/server/publish_decode_diagnostic.sh"


def test_diagnostic_script_exports_bounded_sanitized_failure_evidence(tmp_path):
    project = tmp_path / "project"
    run_id = "20260921T123741-0a4734"
    point = f"{run_id}-zero-offload-profile"
    decision = project / "runs/decode-decision" / run_id
    launcher = project / "runs/decode-mechanism" / f"{point}-launcher"
    decision.mkdir(parents=True)
    launcher.mkdir(parents=True)
    (decision / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "commit": "a" * 40,
                "status": "failed",
                "suite_error_type": "RuntimeError",
                "build": {"status": "complete", "source_sha": "a" * 40},
                "points": {
                    point: {
                        "label": "zero-offload-profile",
                        "status": "failed",
                        "mode": "offload",
                        "error_type": "RuntimeError",
                        "exit_code": 143,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    noisy = "\n".join(f"old-line-{index}" for index in range(100))
    (launcher / "runner.stderr.log").write_text(
        noisy
        + f"\nRuntimeError: failed below {project}\n"
        + "TOKEN=do-not-publish\n"
        + "actual_expert_ids=[1, 2, 3]\n",
        encoding="utf-8",
    )
    (launcher / "runner.stdout.log").write_text(
        "engine initialized\nparent process exited\n", encoding="utf-8"
    )

    run = subprocess.run(
        ["bash", str(SCRIPT), "--run-id", run_id],
        cwd=ROOT,
        env={
            **os.environ,
            "FLEXMOE_DIAGNOSTIC_TESTING": "1",
            "FLEXMOE_DIAGNOSTIC_TEST_ROOT": str(project),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    output = project / f"docs/results/decode-diagnostic-{run_id}.md"
    report = output.read_text(encoding="utf-8")
    assert "zero-offload-profile" in report
    assert "exit_code: `143`" in report
    assert "RuntimeError: failed below $PROJECT_ROOT" in report
    assert "engine initialized" in report
    assert "parent process exited" in report
    assert "do-not-publish" not in report
    assert "actual_expert_ids" not in report
    assert str(project) not in report
    assert "old-line-0" not in report
    assert "old-line-99" in report
    assert run.stdout.strip() == str(output)


def test_diagnostic_script_fails_when_requested_run_is_missing(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    run = subprocess.run(
        ["bash", str(SCRIPT), "--run-id", "missing-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "FLEXMOE_DIAGNOSTIC_TESTING": "1",
            "FLEXMOE_DIAGNOSTIC_TEST_ROOT": str(project),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode != 0
    assert "state.json" in run.stderr
    assert not list(project.glob("docs/results/decode-diagnostic-*.md"))


def test_diagnostic_script_auto_selects_latest_failed_run(tmp_path):
    project = tmp_path / "project"
    failed = project / "runs/decode-decision/failed-run"
    complete = project / "runs/decode-decision/complete-run"
    failed.mkdir(parents=True)
    complete.mkdir(parents=True)
    (failed / "state.json").write_text(
        json.dumps(
            {
                "run_id": "failed-run",
                "commit": "a" * 40,
                "status": "failed",
                "suite_error_type": "RuntimeError",
                "points": {},
            }
        ),
        encoding="utf-8",
    )
    (complete / "state.json").write_text(
        json.dumps(
            {
                "run_id": "complete-run",
                "commit": "a" * 40,
                "status": "complete",
                "points": {},
            }
        ),
        encoding="utf-8",
    )
    os.utime(failed / "state.json", (2, 2))
    os.utime(complete / "state.json", (3, 3))

    run = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "FLEXMOE_DIAGNOSTIC_TESTING": "1",
            "FLEXMOE_DIAGNOSTIC_TEST_ROOT": str(project),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().endswith("decode-diagnostic-failed-run.md")
