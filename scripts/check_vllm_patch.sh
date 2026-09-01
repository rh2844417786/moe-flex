#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="https://github.com/vllm-project/vllm.git"
commit="01efc7ef781391e744ed08c3292817a773d654e6"
patch_file="${project_root}/patches/vllm-v0.10.2.patch"
audit_root="$(mktemp -d "${TMPDIR:-/tmp}/moe-flex-vllm-audit.XXXXXXXX")"

cleanup() {
  rm -rf "${audit_root}"
}
trap cleanup EXIT

git clone --filter=blob:none --no-checkout "${repository}" "${audit_root}/vllm"
git -C "${audit_root}/vllm" checkout --detach "${commit}"
test "$(git -C "${audit_root}/vllm" rev-parse HEAD)" = "${commit}"
git -C "${audit_root}/vllm" apply --unidiff-zero --check "${patch_file}"
git -C "${audit_root}/vllm" apply --unidiff-zero "${patch_file}"
test "$(git -C "${audit_root}/vllm" diff --name-only)" = \
  "vllm/model_executor/layers/fused_moe/layer.py"
