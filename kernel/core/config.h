// SPDX-License-Identifier: Apache-2.0
// Compile-time configuration of the co-tenant workloads: block shape, game
// rules, the policy network and the trainer. The screen size, kSlots and
// kLlmMaxTokens set the shared-memory layout: after changing them, rebuild
// kernel/ and regenerate server/layout.json (deploy/serve.sh does both).
#pragma once

#include <cstdint>

#ifndef COTENANT_THREADS
#define COTENANT_THREADS 256
#endif

namespace cotenant {

// ---------------------------------------------------------------- blocks

constexpr int kThreads = COTENANT_THREADS; // threads per co-tenant block
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

// Which co-tenant workloads run, from $COTENANT_MODE.
enum Mode : int {
  kModeShowcase = 0,     // single-game view + training (benchmarks, videos)
  kModeGameOnly = 1,     // single-game view, no training
  kModeIdle = 2,         // co-tenant blocks exit at once (overhead baseline)
  kModeTrainingOnly = 3, // training, no game
  kModeArcade = 4,       // the arcade (players race the AI) + training
};

// ---------------------------------------------------------------- game
// Distances are in pixels, times in ticks. Mirrors reference/a2c_flappy.py.

constexpr int kScreenW = 288;
constexpr int kScreenH = 512;
constexpr int kGroundY = 404; // top of the ground

constexpr int kBirdX = 57; // left edge of every bird
constexpr int kBirdW = 34;
constexpr int kBirdH = 24;
constexpr float kBirdStartY = kScreenH * 0.35f;
constexpr float kGravity = 1.f;       // added to the bird's speed every tick
constexpr float kFlapSpeed = -9.f;    // speed right after a flap
constexpr float kMaxFallSpeed = 10.f; // also normalizes the observed speed

constexpr int kNumPipes = 3;
constexpr int kPipeW = 52;
constexpr int kPipeGap = 100;     // height of the opening
constexpr int kPipeSpacing = 160; // left edge to left edge
constexpr int kFirstPipeX = kScreenW + 60;
constexpr int kPipeSpeed = 4;     // per tick, leftwards
constexpr int kGapTopMin = 80;    // the gap's top is uniform in
constexpr int kGapTopRange = 142; // [kGapTopMin, kGapTopMin + kGapTopRange)

constexpr float kRewardPerTick = 0.1f; // plus 1 per pipe passed
constexpr float kCrashPenalty = 2.f;

constexpr int kTicksPerSecond = 30;
constexpr long long kTickNs = 1000000000LL / kTicksPerSecond;

// ---------------------------------------------------------------- policy
// kObs inputs -> kHidden -> kHidden -> (flap logit, state value), ReLU.
// Parameters are one flat float array in PyTorch's state_dict order.

constexpr int kObs = 6;
constexpr int kHidden = 64;
constexpr int kHeads = 2; // flap logit, state value

constexpr int kOffW1 = 0;                          // [kHidden][kObs]
constexpr int kOffB1 = kOffW1 + kHidden * kObs;    // [kHidden]
constexpr int kOffW2 = kOffB1 + kHidden;           // [kHidden][kHidden]
constexpr int kOffB2 = kOffW2 + kHidden * kHidden; // [kHidden]
constexpr int kOffW3 = kOffB2 + kHidden;           // [kHeads][kHidden]
constexpr int kOffB3 = kOffW3 + kHeads * kHidden;  // [kHeads]
constexpr int kParams = kOffB3 + kHeads;
constexpr int kParamsPadded = // whole warps, for the copy loops
    (kParams + kWarpSize - 1) / kWarpSize * kWarpSize;

// ---------------------------------------------------------------- trainer

constexpr int kEnvsPerLearner = kThreads; // one environment per thread
constexpr int kRolloutLen = 16;           // steps per environment per update

constexpr float kLearningRate = 1e-3f;
constexpr float kAdamBeta1 = 0.9f;
constexpr float kAdamBeta2 = 0.999f;
constexpr float kAdamEps = 1e-8f;
constexpr float kDiscount = 0.99f;
constexpr float kEntropyCoef = 0.01f;
constexpr float kValueCoef = 0.5f;

// ---------------------------------------------------------------- views

constexpr int kSlots = 8;           // arcade games (slot 0: spectator)
constexpr int kLlmMaxTokens = 8192; // >= the LLM's max sequence length

} // namespace cotenant
