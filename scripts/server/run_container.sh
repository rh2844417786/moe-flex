#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_root="/home/jovyan/wangtonghan/moe-flex"
if [[ "${project_root}" != "${expected_root}" ]]; then
  echo "refusing project root ${project_root}; expected ${expected_root}" >&2
  exit 2
fi
if [[ -z "${GPU_IDS:-}" ]]; then
  echo "GPU_IDS must be an explicit comma-separated list" >&2
  exit 3
fi
if [[ "$#" -eq 0 ]]; then
  echo "usage: GPU_IDS=0,1,2,3 $0 COMMAND [ARG ...]" >&2
  exit 4
fi

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "formal FluxMoE commands require exactly four GPUs" >&2
  exit 5
fi
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU ID ${gpu}" >&2
    exit 6
  fi
  if [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "duplicate GPU ID ${gpu}" >&2
    exit 7
  fi
  seen_gpus["${gpu}"]=1
done

git_sha="$(git -C "${project_root}" rev-parse HEAD)"
image_env="${project_root}/build/image.env"
if [[ ! -s "${image_env}" ]]; then
  echo "run scripts/server/build.sh before starting a container" >&2
  exit 8
fi
source "${project_root}/build/image.env"
if [[ -z "${IMAGE:-}" || -z "${GIT_SHA:-}" ]]; then
  echo "invalid image metadata in ${image_env}" >&2
  exit 9
fi
if [[ "${GIT_SHA}" != "${git_sha}" ]]; then
  echo "image metadata is for ${GIT_SHA}, checkout is ${git_sha}" >&2
  exit 10
fi
image="${IMAGE}"
process_name="wth333"
container_name="wth333-moe-flex-${git_sha:0:12}-$$"
image_revision="$(docker image inspect "${image}" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')"
if [[ "${image_revision}" != "${git_sha}" ]]; then
  echo "build and verify ${image} with scripts/server/build.sh first" >&2
  exit 11
fi

container_gpu_ids=""
for ((index = 0; index < ${#gpu_array[@]}; index++)); do
  if [[ -n "${container_gpu_ids}" ]]; then
    container_gpu_ids+=","
  fi
  container_gpu_ids+="${index}"
done

# This H100 host requires the bonded management interface for NCCL bootstrap
# and disables NVLink Switch collectives, which otherwise stall before load.
nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-bond0}"
nccl_p2p_level="${NCCL_P2P_LEVEL:-NVL}"
nccl_nvls_enable="${NCCL_NVLS_ENABLE:-0}"

# Docker CLI 29 parses an unquoted comma-separated device list as both a count
# and DeviceIDs. Preserve the inner quotes as part of the argument.
exec docker run --rm \
  --name "${container_name}" \
  --entrypoint "" \
  --gpus "\"device=${GPU_IDS}\"" \
  --ipc=host \
  --network=host \
  --ulimit nofile=65535:65535 \
  --mount "type=bind,src=${project_root},dst=${expected_root}" \
  --mount "type=bind,src=/mnt/public_data,dst=/mnt/public_data,readonly" \
  --workdir "${expected_root}" \
  --env "CUDA_VISIBLE_DEVICES=${container_gpu_ids}" \
  --env "HOST_GPU_IDS=${GPU_IDS}" \
  --env "FLEXMOE_PROJECT_ROOT=${expected_root}" \
  --env "FLUXMOE_PROCESS_NAME=${process_name}" \
  --env "NCCL_SOCKET_IFNAME=${nccl_socket_ifname}" \
  --env "NCCL_P2P_LEVEL=${nccl_p2p_level}" \
  --env "NCCL_NVLS_ENABLE=${nccl_nvls_enable}" \
  "${image}" "$@"
