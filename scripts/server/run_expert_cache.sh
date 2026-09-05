#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: GPU_IDS=0,1,2,3 bash scripts/server/run_expert_cache.sh calibrate|confirm|export|point|cuda-check [FLAGS]"
  echo "calibrate: native held-out capture; confirm: equal-budget R/B/C; export: sanitized evidence"
  exit 0
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_root="/home/jovyan/wangtonghan/moe-flex"
if [[ "${project_root}" != "${expected_root}" ]]; then
  echo "refusing execution outside ${expected_root}" >&2
  exit 2
fi
mode="${1:?choose calibrate, confirm, export, point or cuda-check}"
shift
case "${mode}" in
  calibrate|confirm|export)
    exec python3 "${project_root}/src/flexmoe/bench/expert_cache_suite.py" \
      "${mode}" --project-root "${project_root}" "$@"
    ;;
  point|calibrate-point|cuda-check) ;;
  *) echo "unknown command; use --help" >&2; exit 3 ;;
esac

timeout_s=7200
if [[ "${1:-}" == "--timeout-s" ]]; then
  timeout_s="${2:?timeout seconds required}"
  shift 2
fi
if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
  echo "timeout must be positive integer seconds" >&2
  exit 4
fi
if [[ -n "$(git -C "${project_root}" status --porcelain)" ]]; then
  echo "commit code fixes before rebuilding/running; checkout must be clean" >&2
  exit 5
fi

cache_root="${project_root}/build/partial-cache"
mkdir -p "${cache_root}/tmp" "${cache_root}/xdg" "${cache_root}/huggingface" \
  "${cache_root}/vllm" "${cache_root}/torch" "${cache_root}/torch-extensions" \
  "${cache_root}/triton" "${cache_root}/cuda"
offline=(env "TMPDIR=${cache_root}/tmp" "XDG_CACHE_HOME=${cache_root}/xdg" \
  "HF_HOME=${cache_root}/huggingface" "VLLM_CACHE_ROOT=${cache_root}/vllm" \
  "TORCH_HOME=${cache_root}/torch" "TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions" \
  "TRITON_CACHE_DIR=${cache_root}/triton" "CUDA_CACHE_PATH=${cache_root}/cuda" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1)
model_path="/mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[$i]}" == "--model-path" ]]; then
    model_path="${arguments[$((i+1))]:?model path required}"
  fi
done
# Recheck GPU ownership for every fresh engine; never kill another user's jobs.
bash "${project_root}/scripts/server/run_container.sh" "${offline[@]}" \
  timeout --signal=TERM --kill-after=30 "${timeout_s}" \
  python3 -m flexmoe.runtime.preflight check --project-root "${project_root}" \
  --model-path "${model_path}" --gpu-ids 0,1,2,3 \
  --output "${project_root}/runs/expert-preflight-$(git -C "${project_root}" rev-parse HEAD)-$$.json"
if [[ "${mode}" == "cuda-check" ]]; then
  exec bash "${project_root}/scripts/server/run_container.sh" "${offline[@]}" \
    timeout --signal=TERM --kill-after=30 "${timeout_s}" \
    python3 -m flexmoe.bench.expert_cache_preflight --project-root "${project_root}" "$@"
fi
# timeout lives inside the container, so distributed workers cannot outlive it.
exec bash "${project_root}/scripts/server/run_container.sh" "${offline[@]}" \
  timeout --signal=TERM --kill-after=30 "${timeout_s}" \
  python3 -m flexmoe.bench.expert_cache_runner "${mode}" --project-root "${project_root}" "$@"
