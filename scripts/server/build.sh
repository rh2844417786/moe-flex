#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_root="/home/jovyan/wangtonghan/moe-flex"
if [[ "${project_root}" != "${expected_root}" ]]; then
  echo "refusing project root ${project_root}; expected ${expected_root}" >&2
  exit 2
fi

git_sha="$(git -C "${project_root}" rev-parse HEAD)"
base_image="vllm/vllm-openai:v0.10.2@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db"
image="moe-flex-local:${git_sha}"

# Docker Hub is the only required non-Git transport. All RUN instructions are
# forced offline so an accidental apt/PyPI/GitHub dependency fails immediately.
docker pull "${base_image}"
docker build \
  --pull=false \
  --network=none \
  --build-arg "VCS_REF=${git_sha}" \
  --build-arg "VLLM_COMMIT=01efc7ef781391e744ed08c3292817a773d654e6" \
  --tag "${image}" \
  --file "${project_root}/docker/Dockerfile" \
  "${project_root}"

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
printf 'IMAGE=%s\nGIT_SHA=%s\nBASE_IMAGE=%s\n' \
  "${image}" "${git_sha}" "${base_image}" > \
  "${project_root}/build/image.env"
echo "${image}"
