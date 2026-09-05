#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: GPU_IDS=0,1,2,3 bash scripts/server/run_partial_offload.sh scan|confirm|export|point [FLAGS]"
  echo "scan: short candidate search; confirm: selected R/B/C; export: numeric Git return package"
  echo "Use COMMAND --help for flags. Build with scripts/server/build.sh first."
  exit 0
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_root="/home/jovyan/wangtonghan/moe-flex"
if [[ "${project_root}" != "${expected_root}" ]]; then
  echo "refusing project root ${project_root}; expected ${expected_root}" >&2
  exit 2
fi
mode="${1:-scan}"
if [[ "$#" -gt 0 ]]; then shift; fi

if [[ "${mode}" == "point" ]]; then
  timeout_s=1800
  if [[ "${1:-}" == "--timeout-s" ]]; then
    timeout_s="${2:?--timeout-s needs seconds}"
    shift 2
  fi
  if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
    echo "timeout must be positive integer seconds" >&2
    exit 3
  fi
  cache_root="${project_root}/build/partial-cache"
  mkdir -p "${cache_root}/tmp" "${cache_root}/xdg" "${cache_root}/huggingface" \
    "${cache_root}/vllm" "${cache_root}/torch" "${cache_root}/torch-extensions" \
    "${cache_root}/triton" "${cache_root}/cuda"
  # GPU exclusivity and availability can change between points. Never reuse
  # an older successful preflight just because the commit is unchanged.
  bash "${project_root}/scripts/server/preflight.sh"
  # timeout is PID 1 INSIDE the container. When it exits, Docker also stops
  # surviving distributed workers; a killed Docker client cannot strand them.
  exec bash "${project_root}/scripts/server/run_container.sh" \
    env "TMPDIR=${cache_root}/tmp" "XDG_CACHE_HOME=${cache_root}/xdg" \
    "HF_HOME=${cache_root}/huggingface" "VLLM_CACHE_ROOT=${cache_root}/vllm" \
    "TORCH_HOME=${cache_root}/torch" "TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions" \
    "TRITON_CACHE_DIR=${cache_root}/triton" "CUDA_CACHE_PATH=${cache_root}/cuda" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    timeout --signal=TERM --kill-after=30 "${timeout_s}" \
    python3 -m flexmoe.bench.partial_runner --project-root "${project_root}" "$@"
fi

case "${mode}" in
  scan|confirm|export)
    # Run by filename to avoid package __init__ importing torch on the host.
    exec python3 "${project_root}/src/flexmoe/bench/partial_suite.py" \
      "${mode}" --project-root "${project_root}" "$@"
    ;;
  *) echo "unknown mode ${mode}; use --help" >&2; exit 4 ;;
esac
