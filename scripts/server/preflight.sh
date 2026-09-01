#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
git_sha="$(git -C "${project_root}" rev-parse HEAD)"
output="${project_root}/runs/preflight-${git_sha}.json"
mkdir -p "${project_root}/runs"

"${project_root}/scripts/server/run_container.sh" \
  python3 -m flexmoe.runtime.preflight check \
  --project-root "${project_root}" \
  --model-path /mnt/public_data/Qwen/Qwen3-Next-80B-A3B-Instruct \
  --gpu-ids 0,1,2,3 \
  --output "${output}"

echo "${output}"
