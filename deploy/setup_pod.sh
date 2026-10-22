#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copies the repository into a running GPU pod (any
# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel pod with one H100) and runs
# deploy/install.sh there.
# Run from the repository root on your machine: deploy/setup_pod.sh <pod>
# Inside the pod, the repository goes to $FK and Mirage to $MIRAGE.
set -euo pipefail
POD=${1:?usage: deploy/setup_pod.sh <pod>}
FK=${FK:-/root/fk}
MIRAGE=${MIRAGE:-/root/mirage}
kubectl wait --for=condition=Ready "pod/$POD" --timeout=900s
COPYFILE_DISABLE=1 tar cf - VERSION kernel mirage server reference bench deploy |
  kubectl exec -i "$POD" -- bash -c "mkdir -p '$FK' && tar xf - -C '$FK'"
kubectl exec "$POD" -- env MIRAGE="$MIRAGE" bash "$FK/deploy/install.sh"
