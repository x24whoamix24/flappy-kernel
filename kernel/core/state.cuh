// SPDX-License-Identifier: Apache-2.0
// Data shared between the co-tenant blocks and the host:
//   Env, SlotState    game state
//   HostShared        pinned host memory the kernel writes frames and
//                     counters into (its layout is exported for server/)
//   State (g_ct)      device pointers to everything the co-tenants keep
//                     across kernel launches
#pragma once

#include "config.h"

namespace cotenant {

// ---------------------------------------------------------------- game

// A course of pipes and one bird flying through it.
struct Env {
  float y, vy;         // the bird's top edge and vertical speed
  float px[kNumPipes]; // each pipe's left edge
  float gy[kNumPipes]; // top of each pipe's gap
  float score;         // pipes passed by the bird
  uint32_t rng;        // draws the gaps of new pipes
};

// SlotIO::mode, chosen by the player.
enum SlotMode : int { kRace = 0, kSolo = 1, kWatch = 2 };

// SlotIO::phase and SlotState::phase.
enum SlotPhase : int { kIdle = 0, kReady = 1, kPlaying = 2, kOver = 3 };

// One arcade game, in device memory so it survives relaunches. The player's
// bird is the course's own; the AI flies its own bird through the same pipes.
struct SlotState {
  Env course;
  float ai_y, ai_vy, ai_score;
  int you_alive, ai_alive;
  int ai_dead_ticks;
  SlotPhase phase;
  SlotMode mode;
  int start_seen, flap_seen; // last SlotIO::start_seq / flap_seq handled
  int buf;                   // half of the frame buffer last published
  long long tick, frame_seq;
};

// ---------------------------------------------------------------- host block

// One arcade game's controls and status. The host (web server) writes the
// first four fields; the kernel writes the rest.
struct SlotIO {
  int assigned;  // 1 while a player (or the spectator view) uses the slot
  int mode;      // SlotMode
  int start_seq; // bumped by the host to start a new game
  int flap_seq;  // bumped by the host on every flap
  int phase;     // SlotPhase
  int you_score, ai_score, you_alive, ai_alive;
  int frame_buf;
  long long frame_seq;
};

// Lives in pinned host memory mapped into the kernel's address space, so the
// host can watch the game and the counters while the kernel is still running.
// server/app.py reads it by byte offset (`standalone layout`).
struct HostShared {
  // Single-game view (COTENANT_MODE 0/1), double-buffered.
  uint32_t fb[2][kScreenH][kScreenW];
  long long frame_seq;
  int frame_buf;
  int stop; // set by the host to stop the co-tenants
  long long frame_time_ns;
  long long updates, episodes, decode_step;
  float train_mean_score, train_best_score;
  int game_score, game_best, game_deaths;
  // Arcade (COTENANT_MODE=4).
  SlotIO slot[kSlots];
  uint32_t slot_fb[kSlots][2][kScreenH][kScreenW];
  // The LLM request being decoded in this launch, as it is generated:
  // llm_tokens[llm_prompt_len .. llm_pos] are new. llm_launch names the
  // request: its id when Mirage serves online, else a count of launches.
  long long llm_launch;
  int llm_prompt_len, llm_pos;
  long long llm_tokens[kLlmMaxTokens];
};

// ---------------------------------------------------------------- trainer

// One learner's rollout, as structure-of-arrays so that the 32 environments
// of a warp touch 32 consecutive floats.
struct RolloutStep {
  float obs[kObs][kEnvsPerLearner];
  float action[kEnvsPerLearner]; // 1 = flap
  float reward[kEnvsPerLearner]; // replaced in place by the discounted return
  float done[kEnvsPerLearner];   // 1 = the episode ended at this step
};

struct Rollout {
  RolloutStep step[kRolloutLen];
};

// Episodes that finished during one update.
struct EpisodeStats {
  float score_sum, count, best_score;
};

// Written by learner 0 after every update.
struct Progress {
  long long updates;
  long long episodes;
  float best_score;
};

// Grid barrier across the learner blocks. Each field has its own cache line:
// arrivals add to `count` while the waiters poll `generation`.
struct LearnerBarrier {
  alignas(128) unsigned count;
  alignas(128) unsigned generation;
  alignas(128) int stop; // the decision of the last arrival, for everyone
};

// ---------------------------------------------------------------- all state

struct State {
  Mode mode;             // $COTENANT_MODE
  int num_learners;      // co-tenant blocks - 1
  long long max_updates; // $COTENANT_MAX_UPDATES: stop after this many
  int *stop;             // set by the LLM's workers when they terminate
  HostShared *host;      // device alias of the mapped host block

  float *weights; // [kParamsPadded]
  float *adam_m;  // [kParamsPadded]
  float *adam_v;  // [kParamsPadded]
  Progress *progress;

  Env *learner_envs;           // [num_learners * kEnvsPerLearner]
  Rollout *rollouts;           // [num_learners]
  float *partial_grads;        // [num_learners][kParamsPadded]
  EpisodeStats *episode_stats; // [num_learners]
  LearnerBarrier *barrier;

  Env *showcase_env;       // the single-game view's course
  SlotState *slots;        // [kSlots]
  long long *next_tick_ns; // the game block's tick schedule (realtime.cuh)
};

__device__ State g_ct;

} // namespace cotenant
