FROM vllm/vllm-openai:v0.10.2

ARG VCS_REF=unknown
ARG VLLM_COMMIT=01efc7ef781391e744ed08c3292817a773d654e6

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/moe-flex
COPY . /workspace/moe-flex

RUN git clone --filter=blob:none --no-checkout \
      https://github.com/vllm-project/vllm.git /opt/vllm-source \
    && git -C /opt/vllm-source checkout --detach "${VLLM_COMMIT}" \
    && test "$(git -C /opt/vllm-source rev-parse HEAD)" = "${VLLM_COMMIT}" \
    && installed_layer="$(python3 -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/fused_moe/layer.py")')" \
    && cmp -s /opt/vllm-source/vllm/model_executor/layers/fused_moe/layer.py "${installed_layer}" \
    && git -C /opt/vllm-source apply --check /workspace/moe-flex/patches/vllm-v0.10.2.patch \
    && git -C /opt/vllm-source apply /workspace/moe-flex/patches/vllm-v0.10.2.patch \
    && test "$(git -C /opt/vllm-source diff --name-only)" = \
      "vllm/model_executor/layers/fused_moe/layer.py" \
    && install -m 0644 \
      /opt/vllm-source/vllm/model_executor/layers/fused_moe/layer.py \
      "${installed_layer}" \
    && cmp -s /opt/vllm-source/vllm/model_executor/layers/fused_moe/layer.py "${installed_layer}" \
    && python3 -m py_compile "${installed_layer}"

ENV FLEXMOE_BUILD_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV FLEXMOE_VLLM_COMMIT=01efc7ef781391e744ed08c3292817a773d654e6

RUN nvcc --version \
    && python3 -m pip install --disable-pip-version-check --root-user-action=ignore --no-build-isolation -e . \
    && extension_path="$(find /workspace/moe-flex -type f -name '_C*.so' -print -quit)" \
    && test -n "${extension_path}" \
    && test -s "${extension_path}" \
    && file "${extension_path}"

LABEL org.opencontainers.image.source="https://github.com/rh2844417786/moe-flex"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL io.moe-flex.vllm-commit="${VLLM_COMMIT}"
