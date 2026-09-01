#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
git_sha="$(git -C "${project_root}" rev-parse HEAD)"
smoke_success="${project_root}/runs/smoke-${git_sha}/SUCCESS"
correctness_pointer="${project_root}/runs/smoke-${git_sha}/fluxmoe-fixed-run.txt"
if [[ ! -f "${smoke_success}" ]]; then
  echo "current checkout has no successful smoke gate: ${smoke_success}" >&2
  exit 2
fi
if [[ ! -s "${correctness_pointer}" ]]; then
  echo "smoke correctness evidence is missing: ${correctness_pointer}" >&2
  exit 3
fi
correctness_run="$(tr -d '\n' < "${correctness_pointer}")"

points=("32 1024" "128 4096" "256 4096")
pointer_root="${project_root}/build/key-matrix-${git_sha}"
mkdir -p "${pointer_root}"
for point in "${points[@]}"; do
  read -r batch context <<< "${point}"
  resident_pointer="${pointer_root}/resident-${batch}-${context}.txt"
  flux_pointer="${pointer_root}/fluxmoe-fixed-${batch}-${context}.txt"
  "${project_root}/scripts/server/run_container.sh" \
    python3 -m flexmoe.bench.runner \
    --config benchmarks/configs/resident.yaml \
    --project-root "${project_root}" \
    --runs-root "${project_root}/runs" \
    --batch-size "${batch}" \
    --context-length "${context}" \
    --result-path-file "${resident_pointer}"
  resident_run="$(tr -d '\n' < "${resident_pointer}")"

  "${project_root}/scripts/server/run_container.sh" \
    python3 -m flexmoe.bench.runner \
    --config benchmarks/configs/fluxmoe_fixed.yaml \
    --project-root "${project_root}" \
    --runs-root "${project_root}/runs" \
    --batch-size "${batch}" \
    --context-length "${context}" \
    --reference-run "${resident_run}" \
    --correctness-evidence "${correctness_run}" \
    --result-path-file "${flux_pointer}"
  flux_run="$(tr -d '\n' < "${flux_pointer}")"
  "${project_root}/scripts/server/run_container.sh" \
    python3 -m flexmoe.bench.report validate "${flux_run}"
done
