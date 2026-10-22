// SPDX-License-Identifier: Apache-2.0
// What a game frame shows: the course, the birds and the score. The showcase
// and the arcade fill in a Frame each tick; render.cuh draws it.
#pragma once

#include "../core/device_util.cuh"
#include "../core/state.cuh"
#include "font.cuh"

namespace cotenant {

// Nose angle of a bird, in radians, from its vertical speed.
__device__ __forceinline__ float bird_tilt(float vy) {
  return fmaxf(-0.45f, fminf(1.2f, vy * 0.08f));
}
constexpr float kCrashedTilt = 1.2f; // nose down

struct Frame {
  Env course;     // pipes, plus the main bird (the player's in the arcade)
  long long tick; // scrolls the ground, skyline and clouds
  long long seq;  // frames published (single-game view)
  int buf;        // half of the double buffer to draw into
  float main_tilt;
  // The arcade can hide the main bird and add a second, AI-flown one.
  int show_main, show_ai, ai_alive;
  float ai_y, ai_tilt;
  char score[8];
  int score_len;

  __device__ void set_score(double v) {
    score_len = write_count(v, score, int(sizeof(score)));
  }
};

} // namespace cotenant
