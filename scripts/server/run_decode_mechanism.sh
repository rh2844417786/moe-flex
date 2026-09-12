#!/usr/bin/env bash
set -euo pipefail

# Sourceable seams exercise the real argument construction with a container recorder.
decode_clean() {
  local state
  state="$(git -C "$1" status --porcelain -- . ':(exclude)docs/results/decode-mechanism-*')" || return 1
  [[ -z "${state}" ]]
}

decode_paths() {
  python3 -S - "$1" "$2" <<'PY'
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1]).resolve()
name = sys.argv[2]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name) or ".." in name:
    raise SystemExit("fresh safe run ID required")
for relative in (f"runs/decode-mechanism/{name}", f"runs/decode-mechanism/{name}-launcher", f"docs/results/decode-mechanism-{name}"):
    path = root / relative
    if path.resolve() != path or path.exists() or path.is_symlink():
        raise SystemExit("output is not a fresh canonical path")
PY
}

decode_parse() {
  decode_id=""
  decode_timeout=7200
  decode_model=/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct
  decode_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id) decode_id="${2:?run ID required}"; shift 2 ;;
      --timeout-s) decode_timeout="${2:?timeout required}"; shift 2 ;;
      --model-path) decode_model="${2:?model path required}"; decode_args+=("$1" "$2"); shift 2 ;;
      --profile)
        if [[ "${decode_mode}" == native || "${decode_mode}" == calibrate ]]; then
          echo "detailed profiles require matched-resident or offload" >&2; return 3
        fi
        decode_args+=("$1"); shift ;;
      --profile-path|--target-batch|--kv-bytes|--capture-steps|--min-capture-steps|--trace-budget-bytes|--safety-reserve-bytes|--selection-offset|--smoke-output-length)
        if [[ "${decode_mode}" == calibrate ]]; then echo "calibration output and scope are wrapper-owned" >&2; return 3; fi
        decode_args+=("$1" "${2:?flag value required}"); shift 2 ;;
      --dataset-path|--dataset-manifest|--gpu-memory-utilization|--batch-size|--context-length|--output-length|--max-num-seqs|--max-num-batched-tokens|--warmups|--repetitions|--seed|--resident-ratio|--cache-slots|--cache-policy|--calibration-count)
        decode_args+=("$1" "${2:?flag value required}"); shift 2 ;;
      *) echo "unknown, owned or abbreviated runner flag" >&2; return 3 ;;
    esac
  done
  if [[ ! "${decode_timeout}" =~ ^[1-9][0-9]*$ ]]; then return 3; fi
}

decode_container() {
  local cache="${decode_root}/build/partial-cache"
  bash "${decode_root}/scripts/server/run_container.sh" env \
    "TMPDIR=${cache}/tmp" "XDG_CACHE_HOME=${cache}/xdg" \
    "HF_HOME=${cache}/huggingface" "VLLM_CACHE_ROOT=${cache}/vllm" \
    "TORCH_HOME=${cache}/torch" "TORCH_EXTENSIONS_DIR=${cache}/torch-extensions" \
    "TRITON_CACHE_DIR=${cache}/triton" "CUDA_CACHE_PATH=${cache}/cuda" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
    timeout --signal=TERM --kill-after=30 "${decode_timeout}" "$@"
}

decode_run_gpu() {
  local command=()
  if [[ "${decode_mode}" == calibrate ]]; then
    command=(python3 -m flexmoe.bench.expert_cache_runner calibrate-point --profile-path "${decode_run}/profile.json" --timing-samples 0)
  else
    command=(python3 -m flexmoe.bench.decode_mechanism_runner --mode "${decode_mode}")
  fi
  decode_container "${command[@]}" --project-root "${decode_root}" --run-dir "${decode_run}" "${decode_args[@]}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: GPU_IDS=0,1,2,3 bash scripts/server/run_decode_mechanism.sh native|matched-resident|offload|calibrate --run-id ID [--timeout-s N] [exact runner flags]"
  echo "Host: python3 -S src/flexmoe/analysis/decode_suite.py plan|validate|summarize|compare|export --help"
  exit 0
fi
decode_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${decode_root}" != /home/jovyan/wangtonghan/moe-flex ]]; then
  echo "refusing execution outside the authorized server project" >&2; exit 2
fi
decode_mode="${1:?choose native, matched-resident, offload or calibrate}"; shift
case "${decode_mode}" in native|matched-resident|offload|calibrate) ;; *) exit 3 ;; esac
decode_parse "$@"
decode_paths "${decode_root}" "${decode_id}"
if ! decode_clean "${decode_root}"; then
  echo "commit code before running; only generated decode-mechanism public results are exempt" >&2; exit 5
fi
decode_run="${decode_root}/runs/decode-mechanism/${decode_id}"
decode_logs="${decode_root}/runs/decode-mechanism/${decode_id}-launcher"
decode_public="${decode_root}/docs/results/decode-mechanism-${decode_id}"
decode_cache="${decode_root}/build/partial-cache"
mkdir -p "${decode_logs}" "${decode_cache}/tmp" "${decode_cache}/xdg" \
  "${decode_cache}/huggingface" "${decode_cache}/vllm" "${decode_cache}/torch" \
  "${decode_cache}/torch-extensions" "${decode_cache}/triton" "${decode_cache}/cuda"
# The runner creates decode_run exclusively; launcher logs must be its sibling.
decode_code=0
decode_container python3 -m flexmoe.runtime.preflight check --project-root "${decode_root}" \
  --model-path "${decode_model}" --gpu-ids 0,1,2,3 --output "${decode_logs}/preflight.json" \
  >"${decode_logs}/preflight.stdout.log" 2>"${decode_logs}/preflight.stderr.log" || decode_code=$?
if [[ "${decode_code}" -eq 0 ]]; then
  decode_run_gpu >"${decode_logs}/runner.stdout.log" 2>"${decode_logs}/runner.stderr.log" || decode_code=$?
fi
# Exit status lives independently of runner summary/smoke/rejected numeric samples.
python3 -S - "${decode_run}" "${decode_code}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
path = root / "launcher.json"
with path.open("x") as stream:
    json.dump({"status": "complete" if sys.argv[2] == "0" else "failed", "exit_code": int(sys.argv[2])}, stream)
PY
if [[ "${decode_mode}" != calibrate ]]; then
  python3 -S "${decode_root}/src/flexmoe/analysis/decode_suite.py" export \
    --source "${decode_run}" --output "${decode_public}"
fi
exit "${decode_code}"
