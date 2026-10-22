// SPDX-License-Identifier: Apache-2.0
// Game rules: reset, observation, and one physics tick split into the
// pipes and each bird, so the arcade can fly a player and the AI through one
// course. Mirrors reference/a2c_flappy.py exactly.
#pragma once

#include "../core/device_util.cuh"
#include "../core/state.cuh"

namespace cotenant {

__host__ __device__ __forceinline__ float draw_gap_top(uint32_t &rng) {
  return kGapTopMin + floorf(uniform01(rng) * kGapTopRange);
}

// Puts the bird back at the start and lays out a fresh set of pipes.
__host__ __device__ void env_reset(Env &e) {
  e.y = kBirdStartY;
  e.vy = 0.f;
  e.score = 0.f;
  for (int i = 0; i < kNumPipes; i++) {
    e.px[i] = kFirstPipeX + i * kPipeSpacing;
    e.gy[i] = draw_gap_top(e.rng);
  }
}

// True while pipe i's right edge is still ahead of the birds' left edge.
__device__ __forceinline__ bool pipe_ahead(Env const &e, int i) {
  return e.px[i] + kPipeW > kBirdX;
}

// The policy's input for a bird at (y, vy) on course `e`: the bird's height
// and speed, the distance to the nearest pipe ahead, the edges of its gap, and
// the top of the next pipe's gap, normalized by the screen size.
__device__ void env_observe(Env const &e, float y, float vy, float (&o)[kObs]) {
  constexpr float kNone = 1e9f;
  float d0 = kNone, d1 = kNone; // the two nearest pipes ahead
  int i0 = 0, i1 = 0;
  for (int i = 0; i < kNumPipes; i++) {
    float d = pipe_ahead(e, i) ? e.px[i] : kNone;
    if (d < d0) {
      d1 = d0, i1 = i0;
      d0 = d, i0 = i;
    } else if (d < d1) {
      d1 = d, i1 = i;
    }
  }
  if (d1 >= kNone) {
    // Only one pipe ahead: take the next one the way numpy's argsort breaks
    // the tie between the pipes behind.
    for (int i = 0; i < kNumPipes; i++) {
      if (i != i0 && !pipe_ahead(e, i)) {
        i1 = i;
        break;
      }
    }
  }
  o[0] = y / kScreenH;
  o[1] = vy / kMaxFallSpeed;
  o[2] = (e.px[i0] - kBirdX) / kScreenW;
  o[3] = (e.gy[i0] - y) / kScreenH;
  o[4] = (e.gy[i0] + kPipeGap - y) / kScreenH;
  o[5] = (e.gy[i1] - y) / kScreenH;
}

// Moves the pipes one tick, drawing a new gap for each pipe that wraps around.
// Returns how many pipe centres passed the birds' centre (all birds share one
// x).
__device__ float move_pipes(Env &e) {
  constexpr int kBirdCentreX = kBirdX + kBirdW / 2;
  float passed = 0.f;
  for (int i = 0; i < kNumPipes; i++) {
    float centre_before = e.px[i] + kPipeW / 2;
    e.px[i] -= kPipeSpeed;
    float centre_after = e.px[i] + kPipeW / 2;
    if (centre_before >= kBirdCentreX && centre_after < kBirdCentreX) {
      passed += 1.f;
    }
    if (e.px[i] < -kPipeW) {
      e.px[i] += kNumPipes * kPipeSpacing;
      e.gy[i] = draw_gap_top(e.rng);
    }
  }
  return passed;
}

// Moves a bird one tick through the pipes of `e`. Returns false if it crashed
// into a pipe, the ground or the ceiling.
__device__ bool fly_bird(Env const &e, float &y, float &vy, bool flap) {
  vy = flap ? kFlapSpeed : fminf(vy + kGravity, kMaxFallSpeed);
  y += vy;
  for (int i = 0; i < kNumPipes; i++) {
    bool overlap_x = e.px[i] < kBirdX + kBirdW && e.px[i] + kPipeW > kBirdX;
    if (overlap_x && (y < e.gy[i] || y + kBirdH > e.gy[i] + kPipeGap)) {
      return false;
    }
  }
  return !(y + kBirdH >= kGroundY || y < 0.f);
}

// One training step of the course's own bird. Returns the reward; sets `dead`
// and, for a finished episode, its final score (the env is then reset).
__device__ float env_step(Env &e, bool flap, bool &dead, float &final_score) {
  float passed = move_pipes(e);
  dead = !fly_bird(e, e.y, e.vy, flap);
  e.score += passed;
  final_score = e.score;
  if (dead) {
    env_reset(e);
  }
  return kRewardPerTick + passed - (dead ? kCrashPenalty : 0.f);
}

} // namespace cotenant
