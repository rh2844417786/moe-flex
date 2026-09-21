#!/usr/bin/env bash
set -euo pipefail

diagnostic_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${FLEXMOE_DIAGNOSTIC_TESTING:-}" == 1 ]]; then
  diagnostic_root="${FLEXMOE_DIAGNOSTIC_TEST_ROOT:?test root required}"
elif [[ "${diagnostic_root}" != /home/jovyan/wangtonghan/moe-flex ]]; then
  echo "refusing diagnostics outside the authorized server project" >&2
  exit 2
fi

diagnostic_run_id=""
diagnostic_publish=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      diagnostic_run_id="${2:?run ID required}"
      shift 2
      ;;
    --publish)
      diagnostic_publish=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/server/publish_decode_diagnostic.sh [--run-id ID] [--publish]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${diagnostic_run_id}" ]]; then
  diagnostic_run_id="$(python3 -S - "${diagnostic_root}" <<'PY'
import pathlib
import json
import sys

root = pathlib.Path(sys.argv[1])
states = list((root / "runs/decode-decision").glob("*/state.json"))
failed = []
for path in states:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if state.get("status") != "complete":
        failed.append(path)
if not failed:
    raise SystemExit("no failed decode-decision state.json found")
print(max(failed, key=lambda path: path.stat().st_mtime).parent.name)
PY
  )"
fi

if [[ ! "${diagnostic_run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ || "${diagnostic_run_id}" == *..* ]]; then
  echo "invalid run ID" >&2
  exit 2
fi

diagnostic_state="${diagnostic_root}/runs/decode-decision/${diagnostic_run_id}/state.json"
if [[ ! -s "${diagnostic_state}" ]]; then
  echo "missing state.json: ${diagnostic_state}" >&2
  exit 3
fi

diagnostic_output="${diagnostic_root}/docs/results/decode-diagnostic-${diagnostic_run_id}.md"

if [[ "${diagnostic_publish}" -eq 1 ]]; then
  if [[ "$(git -C "${diagnostic_root}" branch --show-current)" != repro/fluxmoe ]]; then
    echo "diagnostic publication requires repro/fluxmoe" >&2
    exit 4
  fi
  if ! git -C "${diagnostic_root}" diff --quiet || ! git -C "${diagnostic_root}" diff --cached --quiet; then
    echo "tracked worktree and index must be clean before diagnostic publication" >&2
    exit 4
  fi
fi

python3 -S - "${diagnostic_root}" "${diagnostic_run_id}" "${diagnostic_output}" <<'PY'
import json
import os
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
run_id = sys.argv[2]
output = pathlib.Path(sys.argv[3])
state_dir = root / "runs/decode-decision" / run_id
state_path = state_dir / "state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
if state.get("run_id") != run_id:
    raise SystemExit("state run ID differs")
if output.resolve().parent != (root / "docs/results").resolve() or output.is_symlink():
    raise SystemExit("diagnostic output must be a canonical project result")

private_payload = re.compile(
    r"prompt_token_ids|output_token_ids|actual_expert_ids|predicted_expert_ids|"
    r"cache_state_before|cache_state_after",
    re.IGNORECASE,
)
secret_assignment = re.compile(
    r"(?i)\b(token|password|secret|api[_-]?key|authorization)\s*[=:]\s*\S+"
)
github_token = re.compile(r"\bgh[opusr]_[A-Za-z0-9_]+\b")
ipv4 = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sanitize(line: str) -> str:
    if private_payload.search(line):
        return "[REDACTED_PRIVATE_PAYLOAD]"
    line = ansi.sub("", line)
    line = line.replace(str(root), "$PROJECT_ROOT")
    line = github_token.sub("<REDACTED_TOKEN>", line)
    line = secret_assignment.sub(lambda match: f"{match.group(1)}=<REDACTED>", line)
    line = ipv4.sub("<REDACTED_IP>", line)
    return line.replace("```", "'''")


def bounded_tail(path: pathlib.Path, lines: int = 60, byte_limit: int = 131072) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size > byte_limit:
            stream.seek(size - byte_limit)
            stream.readline()
        raw = stream.read()
    selected = raw.decode("utf-8", errors="replace").splitlines()[-lines:]
    return "\n".join(sanitize(line) for line in selected).strip()


points = state.get("points") if isinstance(state.get("points"), dict) else {}
build = state.get("build") if isinstance(state.get("build"), dict) else {}
lines = [
    "# Decode decision bounded diagnostic",
    "",
    "> Diagnostic only. Raw routes, prompt/output tokens, weights and full logs are excluded.",
    "",
    f"- Run ID: `{run_id}`",
    f"- Commit: `{state.get('commit', 'unavailable')}`",
    f"- Suite status: `{state.get('status', 'unavailable')}`",
    f"- Suite error: `{state.get('suite_error_type', 'unavailable')}`",
    f"- Build status: `{build.get('status', 'unavailable')}`",
    "",
    "## Point status",
    "",
    "| Point | Label | Mode | Status | Error | Exit code |",
    "|---|---|---|---|---|---:|",
]

problem_points = []
for name, raw in points.items():
    point = raw if isinstance(raw, dict) else {}
    status = point.get("status", "unavailable")
    label = point.get("label", name)
    lines.append(
        f"| `{sanitize(str(name))}` | `{sanitize(str(label))}` | "
        f"`{sanitize(str(point.get('mode', 'unavailable')))}` | `{status}` | "
        f"`{point.get('error_type', '—')}` | `{point.get('exit_code', '—')}` |"
    )
    if status not in ("complete", "skipped"):
        problem_points.append((name, point))

if not points:
    lines.append("| — | — | — | unavailable | — | — |")

lines.extend(("", "## Failure summary", ""))
if problem_points:
    for name, point in problem_points:
        lines.append(
            f"- `{sanitize(str(point.get('label', name)))}`: "
            f"status: `{point.get('status', 'unavailable')}`; "
            f"error: `{point.get('error_type', '—')}`; "
            f"exit_code: `{point.get('exit_code', '—')}`"
        )
else:
    lines.append("- No failed or running point was recorded.")

lines.extend(("", "## Structured preflight", ""))
preflight_written = 0
for name, point in problem_points:
    preflight_path = (
        root / "runs/decode-mechanism" / f"{name}-launcher" / "preflight.json"
    )
    if not preflight_path.is_file() or preflight_path.is_symlink():
        continue
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    checks = preflight.get("checks")
    if not isinstance(checks, list):
        continue
    lines.extend(
        (
            f"### `{sanitize(str(point.get('label', name)))}`",
            "",
            f"- Overall: `{preflight.get('ok') if type(preflight.get('ok')) is bool else 'unavailable'}`",
            "",
            "| Check | OK | Details |",
            "|---|---:|---|",
        )
    )
    for raw in checks[:64]:
        if not isinstance(raw, dict):
            continue
        check_name = sanitize(str(raw.get("name", "unavailable")))[:128]
        ok = raw.get("ok") if type(raw.get("ok")) is bool else "unavailable"
        details = sanitize(str(raw.get("details", ""))).replace("|", "\\|")[:1000]
        lines.append(f"| `{check_name}` | `{ok}` | {details} |")
    lines.append("")
    preflight_written += 1
if not preflight_written:
    lines.append("No structured preflight artifact was available.")

log_candidates = [
    state_dir / "build.stderr.log",
    state_dir / "build.stdout.log",
]
for name, point in problem_points:
    log_candidates.extend(
        (
            state_dir / f"{name}.stderr.log",
            state_dir / f"{name}.stdout.log",
            root / "runs/decode-mechanism" / f"{name}-launcher" / "preflight.stderr.log",
            root / "runs/decode-mechanism" / f"{name}-launcher" / "preflight.stdout.log",
            root / "runs/decode-mechanism" / f"{name}-launcher" / "runner.stderr.log",
            root / "runs/decode-mechanism" / f"{name}-launcher" / "runner.stdout.log",
        )
    )

lines.extend(("", "## Bounded log tails", ""))
seen = set()
written = 0
for path in log_candidates:
    if path in seen:
        continue
    seen.add(path)
    text = bounded_tail(path)
    if not text:
        continue
    relative = path.resolve().relative_to(root)
    lines.extend(
        (
            f"### `{relative}`",
            "",
            f"- Size: `{path.stat().st_size}` bytes; exported tail: at most `60` lines / `131072` bytes.",
            "",
            "```text",
            text,
            "```",
            "",
        )
    )
    written += 1
if not written:
    lines.append("No bounded log files were available.")

output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.{os.getpid()}.new")
temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
temporary.replace(output)
PY

echo "${diagnostic_output}"

if [[ "${diagnostic_publish}" -eq 1 ]]; then
  diagnostic_relative="${diagnostic_output#${diagnostic_root}/}"
  git -C "${diagnostic_root}" add -- "${diagnostic_relative}"
  mapfile -t diagnostic_staged < <(git -C "${diagnostic_root}" diff --cached --name-only)
  if [[ "${#diagnostic_staged[@]}" -ne 1 || "${diagnostic_staged[0]}" != "${diagnostic_relative}" ]]; then
    echo "refusing publication with unexpected staged files" >&2
    exit 5
  fi
  git -C "${diagnostic_root}" commit -m "diagnostics: record decode failure ${diagnostic_run_id}"
  git -C "${diagnostic_root}" push origin HEAD:repro/fluxmoe
fi
