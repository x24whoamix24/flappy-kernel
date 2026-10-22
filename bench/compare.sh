#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Runs the same three workloads (one Qwen3-8B request, A2C training, the game)
# co-located in the megakernel and in the usual separate ways, on one GPU.
# usage: compare.sh [run ...]   (default: all)
# Results go to $RUNS/<run> (default /root/runs); this repository is at $FK.
set -u
FK=${FK:-/root/fk}
RUNS=${RUNS:-/root/runs}
export RUNS
SEP=$FK/kernel/separate
SEQ=${SEQ:-2048}
export PROMPT="Write a long story in eight chapters, at least 3000 words, about a tiny yellow bird that learns, attempt after attempt, to fly through an endless corridor of green pipes. The bird lives inside a GPU, sharing the silicon with a language model that is telling its story and a neural network that is teaching it to fly. Give every chapter a title."
export MPK_THINKING=0

# Trainer and game as their own processes, until $1/stop exists.
# $2: trainer blocks (block 0 idles, the rest are learners). $3/$4: optional
# MPS SM percentages for the trainer and the game.
start_cotenant_procs() {
  local out=$1 blocks=${2:-32}
  rm -f /dev/shm/flappy "$out/ipc" "$out/stop"
  env ${3:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$3} COTENANT_SHM=/flappy COTENANT_MODE=3 \
    $SEP train $blocks "$out/ipc" "$out/stop" > "$out/train.csv" &
  env ${4:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$4} COTENANT_SHM=/flappy COTENANT_OUT=$out \
    $SEP game "$out/ipc" "$out/stop" > "$out/game.log" &
}

stop_cotenant_procs() {
  touch "$1/stop"
  wait
}

run() {
  local name=$1
  case $name in
    cmp_llm_alone | cmp_single128 | cmp_mega | cmp_fused | cmp_streams_128 | cmp_streams_96 | \
      cmp_procs | cmp_mps | cmp_cotenants_alone) ;;
    *) echo "unknown run: $name" >&2; return 1 ;;
  esac
  local out=$RUNS/$name
  rm -rf "$out"; mkdir -p "$out"
  case $name in
    cmp_llm_alone)       $FK/bench/run_mpk.sh $name $SEQ ;;
    cmp_single128)       $FK/bench/run_mpk.sh $name $SEQ MPK_SINGLE_KERNEL=1 ;;
    cmp_mega)            $FK/bench/run_mpk.sh $name $SEQ MPK_COTENANT_BLOCKS=32 MPK_NUM_WORKERS=96 ;;
    cmp_fused)           $FK/bench/run_mpk.sh $name $SEQ MPK_COTENANT_BLOCKS=32 MPK_COTENANT_FUSED=1 MPK_NUM_WORKERS=96 ;;
    cmp_streams_128)     $FK/bench/run_mpk.sh $name $SEQ MPK_COTENANT_BLOCKS=32 MPK_COTENANT_SEPARATE=1 MPK_RESERVED_SMS=0 ;;
    cmp_streams_96)      $FK/bench/run_mpk.sh $name $SEQ MPK_COTENANT_BLOCKS=32 MPK_COTENANT_SEPARATE=1 MPK_NUM_WORKERS=96 ;;
    cmp_procs)
      start_cotenant_procs "$out"
      $FK/bench/run_mpk.sh $name $SEQ
      stop_cotenant_procs "$out" ;;
    cmp_mps)
      # Separate processes under MPS with a manual SM split.
      export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
      mkdir -p /tmp/mps /tmp/mps-log
      nvidia-cuda-mps-control -d
      # 76% = 50 TPCs = 100 SMs for 96 workers + 4 scheduler blocks; the
      # trainer gets 15 TPCs for 30 learners and the game 1 TPC (MPS sizes
      # each client's share in TPCs, and the device reports only that share).
      start_cotenant_procs "$out" 31 24 2
      $FK/bench/run_mpk.sh $name $SEQ CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=76 MPK_NUM_WORKERS=96 MPK_RESERVED_SMS=0
      stop_cotenant_procs "$out"
      echo quit | nvidia-cuda-mps-control
      unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY ;;
    cmp_cotenants_alone)
      start_cotenant_procs "$out"
      sleep 25
      stop_cotenant_procs "$out" ;;
  esac
  echo "done $name"
}

runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(cmp_cotenants_alone cmp_llm_alone cmp_single128 cmp_mega cmp_fused cmp_mps cmp_procs cmp_streams_128 cmp_streams_96)
for r in "${runs[@]}"; do run "$r"; done
echo COMPARE_DONE
