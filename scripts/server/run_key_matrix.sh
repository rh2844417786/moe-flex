#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
git_sha="$(git -C "${project_root}" rev-parse HEAD)"
smoke_success="${project_root}/runs/smoke-${git_sha}/SUCCESS"
if [[ ! -f "${smoke_success}" ]]; then
  echo "current checkout has no successful smoke gate: ${smoke_success}" >&2
  exit 2
fi

points=("32 1024" "128 4096" "256 4096")
configs=("resident" "fluxmoe_fixed")
for point in "${points[@]}"; do
  read -r batch context <<< "${point}"
  for config in "${configs[@]}"; do
    "${project_root}/scripts/server/run_container.sh" \
      python3 -m flexmoe.bench.runner \
      --config "benchmarks/configs/${config}.yaml" \
      --project-root "${project_root}" \
      --runs-root "${project_root}/runs" \
      --batch-size "${batch}" \
      --context-length "${context}"
  done
done
