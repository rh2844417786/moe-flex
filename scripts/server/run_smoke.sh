#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
git_sha="$(git -C "${project_root}" rev-parse HEAD)"
smoke_root="${project_root}/runs/smoke-${git_sha}"
preflight="${project_root}/runs/preflight-${git_sha}.json"
if [[ ! -s "${preflight}" ]]; then
  echo "run scripts/server/preflight.sh first" >&2
  exit 2
fi
if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] else 1)' "${preflight}"; then
  echo "preflight did not pass: ${preflight}" >&2
  exit 3
fi
mkdir -p "${smoke_root}"

VLLM_AUDIT_SOURCE="${VLLM_AUDIT_SOURCE:-}" \
  "${project_root}/scripts/check_vllm_patch.sh"

"${project_root}/scripts/server/run_container.sh" bash -lc '
  set -euo pipefail
  pytest tests/unit -q
  pytest tests/cuda/test_extension_import.py tests/cuda/test_paged_region.py tests/cuda/test_stream_lifecycle.py tests/cuda/test_huffman_cuda.py tests/cuda/test_storage_hierarchy.py -q
  pytest tests/integration/test_vllm_patch.py -q
  pytest tests/integration/test_fused_moe_parity.py -q
  compute-sanitizer --error-exitcode=9 --tool memcheck python3 -m pytest tests/cuda/test_paged_region.py tests/cuda/test_huffman_cuda.py tests/cuda/test_storage_hierarchy.py -q -k "not ten_thousand"
  compute-sanitizer --error-exitcode=10 --tool racecheck python3 -m pytest tests/cuda/test_stream_lifecycle.py -q -k "not ten_thousand"
'

resident_pointer="${smoke_root}/resident-run.txt"
flux_pointer="${smoke_root}/fluxmoe-fixed-run.txt"
# vLLM 0.10.2's early Qwen3-Next hybrid scheduler is not batch invariant.
# Isolate one request so this gate measures resident/FluxMoE parity rather
# than request packing order; the key matrix retains production batches.
"${project_root}/scripts/server/run_container.sh" \
  python3 -m flexmoe.bench.runner \
  --config benchmarks/configs/resident.yaml \
  --project-root "${project_root}" \
  --runs-root "${project_root}/runs" \
  --batch-size 1 \
  --context-length 1024 \
  --correctness-mode \
  --result-path-file "${resident_pointer}"
resident_run="$(tr -d '\n' < "${resident_pointer}")"

"${project_root}/scripts/server/run_container.sh" \
  python3 -m flexmoe.bench.runner \
  --config benchmarks/configs/fluxmoe_fixed.yaml \
  --project-root "${project_root}" \
  --runs-root "${project_root}/runs" \
  --batch-size 1 \
  --context-length 1024 \
  --reference-run "${resident_run}" \
  --correctness-mode \
  --result-path-file "${flux_pointer}"
flux_run="$(tr -d '\n' < "${flux_pointer}")"
"${project_root}/scripts/server/run_container.sh" \
  python3 -m flexmoe.bench.report validate "${flux_run}"

touch "${smoke_root}/SUCCESS"
echo "${smoke_root}"
