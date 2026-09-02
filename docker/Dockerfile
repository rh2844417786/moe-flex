FROM vllm/vllm-openai:v0.10.2@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db

ARG VCS_REF=unknown
ARG VLLM_COMMIT=01efc7ef781391e744ed08c3292817a773d654e6
ARG VLLM_LAYER_SHA256=f6a741cf80028f10c261ccdc72ab54b5301f865cb3a1cc5209656459cb796198
ARG PATCHED_LAYER_SHA256=a79ba498cabc41bc03ca4f2e51b92d54422aa39de1f53032e88e13a012bbcbcd

USER root
WORKDIR /workspace/moe-flex
COPY . /workspace/moe-flex

ENV FLEXMOE_BUILD_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV FLEXMOE_VLLM_COMMIT=01efc7ef781391e744ed08c3292817a773d654e6

# The Docker Hub vLLM image already contains git, the CUDA devel toolchain,
# Ninja, PyTorch 2.8.0, and all runtime dependencies. Every RUN step below is
# valid with `docker build --network=none`.
RUN command -v git \
    && command -v c++ \
    && command -v nvcc \
    && python3 -c 'import ninja, torch; assert torch.__version__.split("+")[0] == "2.8.0"'

RUN installed_root="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && installed_layer="${installed_root}/vllm/model_executor/layers/fused_moe/layer.py" \
    && echo "${VLLM_LAYER_SHA256}  ${installed_layer}" | sha256sum --check --strict \
    && git -C "${installed_root}" apply --unidiff-zero \
      /workspace/moe-flex/patches/vllm-v0.10.2.patch \
    && echo "${PATCHED_LAYER_SHA256}  ${installed_layer}" | sha256sum --check --strict \
    && python3 -m py_compile "${installed_layer}"

RUN cd /workspace/moe-flex \
    && echo "b6a85871a79d2e3b22d2d1b94ac2824226a63c6b741c88f7ae975f18b6778374  docker/wheels/iniconfig-2.0.0-py3-none-any.whl" | sha256sum --check --strict \
    && echo "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c  docker/wheels/packaging-26.3-py3-none-any.whl" | sha256sum --check --strict \
    && echo "44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669  docker/wheels/pluggy-1.5.0-py3-none-any.whl" | sha256sum --check --strict \
    && echo "c69214aa47deac29fad6c2a4f590b9c4a9fdb16a403176fe154b79c0b4d4d820  docker/wheels/pytest-8.3.5-py3-none-any.whl" | sha256sum --check --strict \
    && python3 -m pip install --disable-pip-version-check --root-user-action=ignore \
      --no-index --no-deps --find-links=/workspace/moe-flex/docker/wheels \
      iniconfig==2.0.0 packaging==26.3 pluggy==1.5.0 pytest==8.3.5

RUN nvcc --version \
    && python3 -m pip install --disable-pip-version-check \
      --root-user-action=ignore --no-index --no-deps --no-build-isolation -e . \
    && extension_path="$(find /workspace/moe-flex -type f -name '_C*.so' -print -quit)" \
    && test -n "${extension_path}" \
    && test -s "${extension_path}"

LABEL org.opencontainers.image.source="https://github.com/rh2844417786/moe-flex"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL io.moe-flex.vllm-commit="${VLLM_COMMIT}"
