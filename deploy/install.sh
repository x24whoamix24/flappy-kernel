#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Installs what the arcade needs on a Linux machine with an H100 and CUDA 12.8,
# as root (for example in a pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
# container): system and Python packages, Mirage at the pinned commit with
# mirage/megakernel.patch, and the CUDA tools in kernel/.
#
# Mirage comes prebuilt from the GitHub release when one matches this patch,
# Python, CPU and CUDA version (under a minute); otherwise, or with
# MIRAGE_PREBUILT=0, it is built from source (about 5 minutes on 64 cores).
#
#   deploy/install.sh                  # from anywhere; Mirage goes to $MIRAGE
#   deploy/install.sh --package <dir>  # build from source, then write the
#                                      # release archive and checksum to <dir>
#   FK=<repo> MIRAGE=<dir> deploy/serve.sh
set -euo pipefail
FK=$(cd "$(dirname "$0")/.." && pwd)
MIRAGE=${MIRAGE:-/root/mirage}
MIRAGE_COMMIT=ca58a02d110a278bffaed503be5e66494db2597d
RELEASES=${RELEASES:-https://github.com/x24whoamix24/flappy-kernel/releases/latest/download}
PACKAGE_DIR=
if [ "${1:-}" = --package ]; then
  PACKAGE_DIR=$(mkdir -p "${2:?usage: deploy/install.sh --package <dir>}" && cd "$2" && pwd)
  MIRAGE_PREBUILT=0
fi

apt-get update -qq >/dev/null
apt-get install -y -qq git build-essential ffmpeg curl fonts-dejavu-core >/dev/null
pip install -q -r "$FK/server/requirements.txt"

# The prebuilt archive's name says what it was built from and for; a changed
# patch or another Python, CPU or CUDA version names an archive that does not
# exist, so the install builds from source.
patch_id=$(sha256sum "$FK/mirage/megakernel.patch" | cut -c1-12)
py_tag=$(python -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')
cuda_tag=cu$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
ASSET=mirage-${MIRAGE_COMMIT:0:7}-$patch_id-linux-$(uname -m)-$py_tag-$cuda_tag.tar.gz

# Downloads the prebuilt archive and its checksum into $1; fails when the
# release has no archive of that name.
download_prebuilt() {
  [ "${MIRAGE_PREBUILT:-1}" = 1 ] && [ ! -e "$MIRAGE" ] &&
    curl -fsSL -o "$1/mirage.tar.gz" "$RELEASES/$ASSET" &&
    curl -fsSL -o "$1/mirage.sha256" "$RELEASES/$ASSET.sha256"
}

build_from_source() {
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
}

download=$(mktemp -d)
if download_prebuilt "$download"; then
  (cd "$download" && echo "$(cut -d' ' -f1 mirage.sha256)  mirage.tar.gz" | sha256sum -c --quiet) || {
    echo "$ASSET does not match its checksum" >&2
    exit 1
  }
  mkdir -p "$MIRAGE"
  tar xzf "$download/mirage.tar.gz" -C "$MIRAGE"
  pip install -q -r "$MIRAGE/requirements.txt"
  # What `pip install -e` would register: the package is imported from the tree,
  # whose include/ and deps/ the megakernel is compiled against.
  echo "$MIRAGE/python" >"$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/mirage.pth"
  echo "installed the prebuilt $ASSET"
else
  build_from_source
fi
rm -rf "$download"

if [ -n "$PACKAGE_DIR" ]; then
  # What the arcade runs from: the package (with its native libraries, already
  # copied into python/mirage), the headers and deps the megakernel compiles
  # against, the sources, demo/ and the licenses. Git metadata, build
  # intermediates and Mirage's own tests, benchmarks and tooling stay out.
  tar czf "$PACKAGE_DIR/$ASSET" -C "$MIRAGE" --exclude=.git --exclude=__pycache__ \
    --exclude='*.egg-info' --exclude=./build --exclude=./build.log \
    --exclude=./.claude --exclude=./.github --exclude=./.tools --exclude=./benchmark \
    --exclude=./conda --exclude=./cpp_examples --exclude=./docker --exclude=./docker-build \
    --exclude=./docs --exclude=./img --exclude=./scripts --exclude=./tests .
  (cd "$PACKAGE_DIR" && sha256sum "$ASSET" >"$ASSET.sha256")
  echo "wrote $PACKAGE_DIR/$ASSET"
fi

make -C "$FK/kernel"
echo "installed; start the arcade with: FK=$FK MIRAGE=$MIRAGE $FK/deploy/serve.sh"
