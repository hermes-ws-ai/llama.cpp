# llama.cpp with DFlash support — multi-stage CUDA build
# Usage on Laguna 2.1: docker run --gpus all -v /path/to/models:/models ghcr.io/hermes-ws-ai/llama.cpp:laguna-dflash

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ── Build stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG CUDA_DOCKER_ARCH=
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH:-native}

WORKDIR /build

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN set -x && \
    cmake -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_STATIC=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_APP=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        && cmake --build build --target llama-server llama-cli llama-quantize -- -j$(nproc)

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

LABEL org.opencontainers.image.source="https://github.com/hermes-ws-ai/llama.cpp"
LABEL org.opencontainers.image.description="llama.cpp with DFlash speculative decoding support for Laguna models"
LABEL org.opencontainers.image.version="laguna-2.1-dflash"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/build/bin/ /usr/local/bin/

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

EXPOSE 8080

ENTRYPOINT ["llama-server"]
CMD ["--help"]
