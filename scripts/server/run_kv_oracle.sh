#!/usr/bin/env bash
set -euo pipefail
oracle_clean_code() {
  local oracle_status
  oracle_status="$(git -C "$1" status --porcelain -- . ':(exclude)docs/results/kv-oracle-*')" || return 1
  [[ -z "${oracle_status}" ]]
}
# Sourceable guard also supports direct behavioral tests against real Git repos.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: GPU_IDS=0,1,2,3 bash scripts/server/run_kv_oracle.sh screen|confirm|export|point [FLAGS]"
  echo "Native KV diagnostic only; no formal offload gain or automatic offloading."
  exit 0
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${project_root}" != "/home/jovyan/wangtonghan/moe-flex" ]]; then
  echo "refusing execution outside the authorized server project" >&2
  exit 2
fi
mode="${1:?choose screen, confirm, export or point}"
shift
case "${mode}" in
  screen|confirm|export)
    exec python3 -S "${project_root}/src/flexmoe/bench/kv_oracle_suite.py" \
      "${mode}" --project-root "${project_root}" "$@" ;;
  point) ;;
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
# Same-SHA screen/confirm runs may have earlier generated diagnostic exports.
# Only this generated results namespace is exempt; runtime/code changes are not.
if ! oracle_clean_code "${project_root}"; then
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
# Recheck ownership before each fresh engine. Never stop another user's process.
bash "${project_root}/scripts/server/run_container.sh" "${offline[@]}" \
  timeout --signal=TERM --kill-after=30 "${timeout_s}" \
  python3 -m flexmoe.runtime.preflight check --project-root "${project_root}" \
  --model-path "${model_path}" --gpu-ids 0,1,2,3 \
  --output "${project_root}/runs/kv-oracle-preflight-$(git -C "${project_root}" rev-parse HEAD)-$$.json"
# The timeout is INSIDE Docker; after termination Docker removes this process tree.
exec bash "${project_root}/scripts/server/run_container.sh" "${offline[@]}" \
  timeout --signal=TERM --kill-after=30 "${timeout_s}" \
  python3 -m flexmoe.bench.kv_oracle_runner --project-root "${project_root}" "$@"
