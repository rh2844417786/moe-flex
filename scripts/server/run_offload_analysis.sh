#!/usr/bin/env bash
set -euo pipefail

analysis_clean_code() {
  local analysis_status
  analysis_status="$(git -C "$1" status --porcelain -- . ':(exclude)docs/results/offload-analysis-*')" || return 1
  [[ -z "${analysis_status}" ]]
}

# Sourceable execution seam: tests supply a command recorder at run_container.sh,
# while production always resolves the fixed server checkout before calling it.
analysis_container() {
  local analysis_root="$1" analysis_timeout="$2"
  shift 2
  local analysis_cache="${analysis_root}/build/partial-cache"
  bash "${analysis_root}/scripts/server/run_container.sh" env \
    "TMPDIR=${analysis_cache}/tmp" "XDG_CACHE_HOME=${analysis_cache}/xdg" \
    "HF_HOME=${analysis_cache}/huggingface" "VLLM_CACHE_ROOT=${analysis_cache}/vllm" \
    "TORCH_HOME=${analysis_cache}/torch" "TORCH_EXTENSIONS_DIR=${analysis_cache}/torch-extensions" \
    "TRITON_CACHE_DIR=${analysis_cache}/triton" "CUDA_CACHE_PATH=${analysis_cache}/cuda" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
    timeout --signal=TERM --kill-after=30 "${analysis_timeout}" "$@"
}

analysis_output_path() {
  python3 -S - "$1" "$2" "$3" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
mode = sys.argv[2]
path = pathlib.Path(sys.argv[3])
path = (path if path.is_absolute() else root / path).resolve()
base = root / ("docs/results" if mode == "export" else "runs/offload-analysis")
try:
    relative = path.relative_to(base)
except ValueError:
    raise SystemExit("output escapes the approved namespace")
if not relative.parts or (mode == "export" and not relative.parts[0].startswith("offload-analysis-")):
    raise SystemExit("output does not name a generated analysis artifact")
PY
}

analysis_parse_gpu_args() {
  analysis_timeout=7200
  analysis_id=""
  analysis_engine=native
  analysis_model=/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct
  analysis_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id) analysis_id="${2:?run ID required}"; shift 2 ;;
      --timeout-s) analysis_timeout="${2:?timeout required}"; shift 2 ;;
      --engine-mode) analysis_engine="${2:?engine mode required}"; shift 2 ;;
      --model-path)
        analysis_model="${2:?model path required}"
        if [[ "${analysis_mode}" != "transport" ]]; then analysis_args+=("$1" "$2"); fi
        shift 2 ;;
      --run-dir|--run-dir=*|--project-root|--project-root=*|--mode|--mode=*|--output|--output=*)
        echo "wrapper owns root, run directory, mode and output" >&2; return 3 ;;
      --synthetic-context) analysis_args+=("$1"); shift ;;
      --dataset-path|--dataset-manifest|--gpu-memory-utilization|--batch-size|--context-length|--output-length|--max-num-seqs|--max-num-batched-tokens|--warmups|--repetitions|--seed|--smoke-output-length|--timing-samples|--max-trace-steps|--trace-budget-bytes|--safety-reserve-bytes|--selection-offset|--exclude-trace|--expert-bytes|--experts-per-batch|--modes|--contention|--iterations|--memory-cap-bytes|--gemm-size)
        analysis_args+=("$1" "${2:?flag value required}"); shift 2 ;;
      *) echo "unknown or abbreviated runner flag" >&2; return 3 ;;
    esac
  done
}

analysis_run_gpu() {
  local analysis_command=()
  if [[ "${analysis_mode}" == "transport" ]]; then
    analysis_command=(python3 -m flexmoe.bench.transfer_microbench --timeout-s "${analysis_timeout}")
  else
    local runner_engine="${analysis_engine}"
    if [[ "${analysis_mode}" == "trace" ]]; then runner_engine=trace; fi
    analysis_command=(python3 -m flexmoe.bench.analysis_runner --mode "${runner_engine}")
  fi
  analysis_container "${analysis_root}" "${analysis_timeout}" "${analysis_command[@]}" \
    --project-root "${analysis_root}" --run-dir "${analysis_run}" "${analysis_args[@]}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: GPU_IDS=0,1,2,3 bash scripts/server/run_offload_analysis.sh resident|trace|transport --run-id ID [--timeout-s N] [runner flags]"
  echo "Host commands: validate|replay|analyze|plan|export; raw output uses runs/offload-analysis, export uses docs/results/offload-analysis-ID."
  exit 0
fi
analysis_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${analysis_root}" != "/home/jovyan/wangtonghan/moe-flex" ]]; then
  echo "refusing execution outside the authorized server project" >&2
  exit 2
fi
analysis_mode="${1:?choose resident, trace, transport or host CLI command}"
shift
analysis_cli="${analysis_root}/src/flexmoe/analysis/cli.py"
case "${analysis_mode}" in
  validate|replay|analyze|plan|export)
    # Raw plan/replay/analysis stays private; only sanitized export is public.
    analysis_args=("$@")
    for ((i=0; i<${#analysis_args[@]}; i++)); do
      if [[ "${analysis_args[$i]}" == "--output" ]]; then
        analysis_output="${analysis_args[$((i+1))]:?output required}"
        analysis_output_path "${analysis_root}" "${analysis_mode}" "${analysis_output}" || exit 3
      elif [[ "${analysis_args[$i]}" == --output=* ]]; then
        echo "use separate --output PATH arguments" >&2; exit 3
      fi
    done
    cd "${analysis_root}"
    exec python3 -S "${analysis_cli}" "${analysis_mode}" "$@" ;;
  resident|trace|transport) ;;
  *) echo "unknown mode" >&2; exit 3 ;;
esac
analysis_parse_gpu_args "$@"
if [[ ! "${analysis_id}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$ || "${analysis_id}" == *".."* ]]; then
  echo "fresh safe --run-id required" >&2; exit 4
fi
if [[ ! "${analysis_timeout}" =~ ^[1-9][0-9]*$ || ! "${analysis_engine}" =~ ^(native|eager)$ ]]; then exit 4; fi
if ! analysis_clean_code "${analysis_root}"; then
  echo "commit code changes before rebuilding/running; only generated offload-analysis results are exempt" >&2; exit 5
fi
analysis_run="${analysis_root}/runs/offload-analysis/${analysis_id}"
analysis_logs="${analysis_root}/runs/offload-analysis/${analysis_id}-launcher"
analysis_output="${analysis_root}/docs/results/offload-analysis-${analysis_id}"
analysis_output_path "${analysis_root}" resident "${analysis_run}" || exit 3
analysis_output_path "${analysis_root}" resident "${analysis_logs}" || exit 3
analysis_output_path "${analysis_root}" export "${analysis_output}" || exit 3
if [[ -e "${analysis_run}" || -e "${analysis_logs}" || -e "${analysis_output}" ]]; then
  echo "run ID already exists; use a fresh ID" >&2; exit 6
fi
analysis_cache="${analysis_root}/build/partial-cache"
mkdir -p "${analysis_cache}/tmp" "${analysis_cache}/xdg" "${analysis_cache}/huggingface" \
  "${analysis_cache}/vllm" "${analysis_cache}/torch" "${analysis_cache}/torch-extensions" \
  "${analysis_cache}/triton" "${analysis_cache}/cuda" "${analysis_logs}"
# No mkdir of analysis_run: the shared runner owns its exclusive creation.
analysis_code=0
analysis_container "${analysis_root}" "${analysis_timeout}" \
  python3 -m flexmoe.runtime.preflight check --project-root "${analysis_root}" \
  --model-path "${analysis_model}" --gpu-ids 0,1,2,3 \
  --output "${analysis_logs}/preflight.json" \
  >"${analysis_logs}/preflight.stdout.log" 2>"${analysis_logs}/preflight.stderr.log" || analysis_code=$?
if [[ "${analysis_code}" -eq 0 ]]; then
  analysis_run_gpu \
    >"${analysis_logs}/runner.stdout.log" 2>"${analysis_logs}/runner.stderr.log" || analysis_code=$?
fi
if [[ "${analysis_code}" -ne 0 ]]; then
  python3 -S "${analysis_cli}" record-failure --run-dir "${analysis_run}" --exit-code "${analysis_code}"
fi
python3 -S "${analysis_cli}" export --source "${analysis_run}" --output "${analysis_output}"
exit "${analysis_code}"
