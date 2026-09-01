FROM vllm/vllm-openai:v0.10.2

ARG VCS_REF=unknown

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/moe-flex
COPY . /workspace/moe-flex

ENV FLEXMOE_BUILD_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=9.0

RUN nvcc --version \
    && python3 -m pip install --disable-pip-version-check --root-user-action=ignore --no-build-isolation -e . \
    && extension_path="$(find /workspace/moe-flex -type f -name '_C*.so' -print -quit)" \
    && test -n "${extension_path}" \
    && test -s "${extension_path}" \
    && file "${extension_path}"

LABEL org.opencontainers.image.source="https://github.com/rh2844417786/moe-flex"
LABEL org.opencontainers.image.revision="${VCS_REF}"
