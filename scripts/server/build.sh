#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_root="/home/jovyan/wangtonghan/moe-flex"
if [[ "${project_root}" != "${expected_root}" ]]; then
  echo "refusing project root ${project_root}; expected ${expected_root}" >&2
  exit 2
fi

git_sha="$(git -C "${project_root}" rev-parse HEAD)"
image="ghcr.io/rh2844417786/moe-flex:${git_sha}"
docker pull "${image}"

image_revision="$(docker image inspect "${image}" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')"
vllm_commit="$(docker image inspect "${image}" --format '{{ index .Config.Labels "io.moe-flex.vllm-commit" }}')"
if [[ "${image_revision}" != "${git_sha}" ]]; then
  echo "image revision ${image_revision} does not match checkout ${git_sha}" >&2
  exit 3
fi
if [[ "${vllm_commit}" != "01efc7ef781391e744ed08c3292817a773d654e6" ]]; then
  echo "unexpected vLLM commit label ${vllm_commit}" >&2
  exit 4
fi

mkdir -p "${project_root}/build"
printf 'IMAGE=%s\nGIT_SHA=%s\n' "${image}" "${git_sha}" > \
  "${project_root}/build/image.env"
echo "${image}"
