#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Installs what the arcade needs on a Linux machine with an H100 and CUDA 12.8,
# as root (for example in a pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
# container): system and Python packages, Mirage at the pinned commit with
# mirage/megakernel.patch, and the CUDA tools in kernel/.
#
#   deploy/install.sh            # from anywhere; Mirage goes to $MIRAGE
#   FK=<repo> MIRAGE=<dir> deploy/serve.sh
set -euo pipefail
FK=$(cd "$(dirname "$0")/.." && pwd)
MIRAGE=${MIRAGE:-/root/mirage}
MIRAGE_COMMIT=ca58a02d110a278bffaed503be5e66494db2597d

apt-get update -qq >/dev/null
apt-get install -y -qq git build-essential ffmpeg curl fonts-dejavu-core >/dev/null
pip install -q -r "$FK/server/requirements.txt"
# Mirage's build compiles two Rust crates.
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
. "$HOME/.cargo/env"

[ -d "$MIRAGE" ] || git clone -q --recursive https://github.com/mirage-project/mirage.git "$MIRAGE"
cd "$MIRAGE"
git checkout -q "$MIRAGE_COMMIT"
git submodule update -q --init --recursive
# Apply the patch once; an already patched checkout is left as it is.
if ! git apply --reverse --check "$FK/mirage/megakernel.patch" 2>/dev/null; then
  git apply "$FK/mirage/megakernel.patch" || {
    echo "Mirage at $MIRAGE has other changes (an older patch?); use a fresh MIRAGE directory or reset it" >&2
    exit 1
  }
fi
echo "building Mirage (log: $MIRAGE/build.log)"
MAX_JOBS=${MAX_JOBS:-64} pip install -q -e . >"$MIRAGE/build.log" 2>&1
make -C "$FK/kernel"
echo "installed; start the arcade with: FK=$FK MIRAGE=$MIRAGE $FK/deploy/serve.sh"
