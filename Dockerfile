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

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Mirage at the pinned commit with the patch (prebuilt from the GitHub release when
# one matches, built from source otherwise; neither needs a GPU), the server's
# Python packages and the CUDA tools in kernel/.
COPY VERSION /opt/flappy-kernel/VERSION
COPY mirage /opt/flappy-kernel/mirage
COPY deploy/install.sh /opt/flappy-kernel/deploy/install.sh
COPY server/requirements.txt /opt/flappy-kernel/server/requirements.txt
COPY kernel /opt/flappy-kernel/kernel
RUN MIRAGE=/opt/mirage /opt/flappy-kernel/deploy/install.sh \
 && rm -rf /var/lib/apt/lists/* /opt/mirage/build/*/CMakeFiles

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
