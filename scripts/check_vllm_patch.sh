#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository="https://github.com/vllm-project/vllm.git"
commit="01efc7ef781391e744ed08c3292817a773d654e6"
patch_file="${project_root}/patches/vllm-v0.10.2.patch"
audit_root="$(mktemp -d "${TMPDIR:-/tmp}/moe-flex-vllm-audit.XXXXXXXX")"
trap 'rm -rf "${audit_root}"' EXIT
if [[ -n "${VLLM_AUDIT_SOURCE:-}" ]]; then
  # Audit a disposable clone so this command never alters the pinned checkout.
  git clone --shared --no-checkout "${VLLM_AUDIT_SOURCE}" "${audit_root}/vllm"
  source_root="${audit_root}/vllm"
  git -C "${source_root}" checkout --detach "${commit}"
else
  git clone --depth 1 --branch v0.10.2 "${repository}" "${audit_root}/vllm"
  source_root="${audit_root}/vllm"
fi

test "$(git -C "${source_root}" rev-parse HEAD)" = "${commit}"
git -C "${source_root}" apply --unidiff-zero --check "${patch_file}"
git -C "${source_root}" apply --unidiff-zero "${patch_file}"
test "$(git -C "${source_root}" diff --name-only)" = \
  "vllm/model_executor/layers/fused_moe/layer.py"
