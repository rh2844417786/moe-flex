#!/usr/bin/env bash
set -euo pipefail

decision_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${decision_root}" != /home/jovyan/wangtonghan/moe-flex ]]; then
  echo 'refusing execution outside the authorized server project' >&2
  exit 2
fi
if [[ "${GPU_IDS:-}" != 0,1,2,3 ]]; then
  echo 'requires explicit GPU_IDS=0,1,2,3 and exclusive idle GPUs' >&2
  exit 2
fi
exec python3 -S "${decision_root}/src/flexmoe/analysis/decode_decision.py" \
  --project-root "${decision_root}" "$@"
