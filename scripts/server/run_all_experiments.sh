#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
command=run
if [[ "${1:-}" == run || "${1:-}" == status ]]; then command="$1"; shift; fi
arguments=("$@")
has_id=0
resume=0
for argument in "${arguments[@]}"; do
  case "${argument}" in
    --run-id|--run-id=*) has_id=1 ;;
    --resume) resume=1 ;;
  esac
done
if [[ "${has_id}" -eq 0 && "${command}" == run ]]; then
  if [[ "${resume}" -eq 1 ]]; then
    echo "resume requires the original --run-id" >&2; exit 2
  fi
  run_id="$(python3 -S -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:6])')"
  arguments+=(--run-id "${run_id}")
  echo "Suite ID: ${run_id}"
fi
exec python3 -S "${project_root}/src/flexmoe/analysis/experiment_autorun.py" \
  "${command}" "${arguments[@]}"
