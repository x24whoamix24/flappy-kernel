#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Runs Mirage's Qwen3-8B MPK demo once and logs its results under $RUNS/<name>
# (default /root/runs). Mirage is at $MIRAGE, this repository at $FK.
# usage: run_mpk.sh <name> <max_seq_len> [ENV=value ...]
#   MPK_COTENANT_BLOCKS=N  append N co-tenant blocks (1 game + N-1 trainers)
#   MPK_NUM_WORKERS=W      LLM worker blocks (default 128)
#   MPK_SINGLE_KERNEL=1    single persistent kernel without co-tenants
#   MPK_RESERVED_SMS=N     leave N SMs unused when sizing schedulers
#   IGNORE_EOS=0           stop at EOS instead of decoding to max_seq_len
name=$1; seq=$2; shift 2
out=${RUNS:-/root/runs}/$name; mkdir -p "$out"
eos_flag=$([ "${IGNORE_EOS:-1}" = "1" ] && echo --ignore-eos)
cd "${MIRAGE:-/root/mirage}/demo/qwen3" || exit 1
env "$@" COTENANT_OUT="$out" MPK_COTENANT_DIR="${FK:-/root/fk}/kernel" MPK_DUMP_RUN="$out/run.json" \
  python demo.py --use-mirage --model Qwen/Qwen3-8B --max-seq-length "$seq" $eos_flag \
  --prompt "${PROMPT:-Give me a short introduction to large language model.}" > "$out/log.txt" 2>&1
echo "$name rc=$? $(grep -E "per-token" "$out/log.txt") $(grep -E "^\[cotenant\]" "$out/log.txt")"
