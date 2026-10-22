// SPDX-License-Identifier: Apache-2.0
// Co-tenant workloads that run inside Mirage's persistent megakernel, next to
// the LLM's decode workers. Mirage's kernel calls cotenant::run() from each
// co-tenant block and cotenant::signal_stop() when its workers terminate; the
// host calls host_init(), host_start() and host_stop() (host/host.h).
//
//   block 0      the arcade (COTENANT_MODE=4) or the single-game showcase
//   blocks 1..N  A2C learners
//
// Layout (includes are relative to this directory):
//   core/config.h          compile-time settings
//   core/device_util.cuh   random numbers, clock, memory ordering, warps
//   core/state.cuh         data shared between blocks and with the host
//   training/policy.cuh    the policy network and its gradient
//   training/trainer.cuh   the learner blocks
//   game/physics.cuh       game rules and observation
//   game/font.cuh          5x7 digits and number formatting
//   game/frame.cuh         what a frame shows (birds and score)
//   game/render.cuh        drawing a frame, layer by layer
//   game/realtime.cuh      the 30 Hz tick of the game block
//   game/showcase.cuh      the single-game view
//   game/arcade.cuh        the arcade: players racing the AI
//   host/host.h            host-side setup, IPC and recording
//   tools/                 standalone and separate-process test tools
// The physics, observation, network and update mirror reference/a2c_flappy.py.
#pragma once

#include "game/arcade.cuh"
#include "game/showcase.cuh"
#include "host/host.h"
#include "training/trainer.cuh"

namespace cotenant {

#ifdef MPK_MAX_SEQ_LENGTH
static_assert(kLlmMaxTokens >= MPK_MAX_SEQ_LENGTH,
              "the LLM token feed must hold the longest sequence");
#endif

// Entry point for each co-tenant block; `ct_block` is in [0, num_blocks).
// `decode_step`, `prompt_length` and `tokens` describe the LLM's current
// request, or are null without an LLM. `request_id` is null unless Mirage
// serves every request in one launch (online); see LlmFeed.
__device__ void run(int const *decode_step,
                    int const *prompt_length,
                    long long const *tokens,
                    int const *request_id,
                    int ct_block,
                    char *smem) {
  if (blockDim.x != kThreads) {
    __trap(); // every index below assumes kThreads threads per block
  }
  Mode const mode = g_ct.mode;
  bool const game_block = ct_block == 0;
  if (mode == kModeIdle || (mode == kModeGameOnly && !game_block) ||
      (mode == kModeTrainingOnly && game_block)) {
    return;
  }
  if (!game_block) {
    run_learner(ct_block - 1, smem);
  } else if (mode == kModeArcade) {
    run_arcade(LlmView{decode_step, prompt_length, tokens, request_id}, smem);
  } else {
    run_showcase(decode_step, smem);
  }
}

// Called by Mirage's workers when they receive TASK_TERMINATE.
__device__ __forceinline__ void signal_stop() {
  st_volatile(g_ct.stop, 1);
  __threadfence();
}

} // namespace cotenant
