# SPDX-License-Identifier: Apache-2.0
# Flappy Kernel: Mirage (with mirage/megakernel.patch), the CUDA co-tenant tools and
# the web server, ready to serve on an NVIDIA H100.
#
#   docker build -t flappy-kernel .
#   docker run --gpus all --shm-size 16g -p 8080:8080 \
#     -v hf-cache:/root/.cache/huggingface flappy-kernel
#
# Model weights are not in the image: the first start downloads Qwen3-8B (16 GB) into
# the mounted cache, then compiles the megakernel (2-3 minutes).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG MIRAGE_REPO=https://github.com/mirage-project/mirage.git
ARG MIRAGE_COMMIT=ca58a02d110a278bffaed503be5e66494db2597d
ARG CUDA_ARCH=sm_90a
ARG MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/root/.cargo/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Mirage's build compiles two Rust crates.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Mirage at the pinned commit, patched. Its CUDA libraries build for sm_75..sm_90 and
# need no GPU at build time; the megakernel itself is compiled when the server starts.
RUN git clone --recursive "$MIRAGE_REPO" /opt/mirage \
 && cd /opt/mirage \
 && git checkout -q "$MIRAGE_COMMIT" \
 && git submodule update -q --init --recursive
COPY mirage/megakernel.patch /opt/flappy-kernel/mirage/megakernel.patch
RUN cd /opt/mirage \
 && git apply /opt/flappy-kernel/mirage/megakernel.patch \
 && MAX_JOBS=$MAX_JOBS pip install -e . \
 && rm -rf /opt/mirage/build/*/CMakeFiles

COPY server/requirements.txt /opt/flappy-kernel/server/requirements.txt
RUN pip install -r /opt/flappy-kernel/server/requirements.txt

COPY kernel /opt/flappy-kernel/kernel
RUN make -C /opt/flappy-kernel/kernel ARCH=$CUDA_ARCH

COPY server /opt/flappy-kernel/server
COPY deploy/serve.sh /opt/flappy-kernel/deploy/serve.sh
COPY LICENSE NOTICE /opt/flappy-kernel/

ENV FK=/opt/flappy-kernel \
    MIRAGE=/opt/mirage \
    SERVE_DIR=/var/lib/flappy-kernel \
    HF_HOME=/root/.cache/huggingface
EXPOSE 8080
WORKDIR /opt/flappy-kernel
CMD ["/opt/flappy-kernel/deploy/serve.sh"]
