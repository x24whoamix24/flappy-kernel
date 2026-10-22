// SPDX-License-Identifier: Apache-2.0
// The single-game view used by the benchmarks and videos: one block plays the
// game in real time with the newest weights and renders each frame to host
// memory. tools/separate.cu also launches showcase_frame() one frame at a
// time.
#pragma once

#include "../core/device_util.cuh"
#include "../training/policy.cuh"
#include "physics.cuh"
#include "realtime.cuh"
#include "render.cuh"

namespace cotenant {

// Thread 0: advances the game one tick with the greedy policy and updates
// the frame's bird and score and the host's counters.
__device__ void
showcase_tick(Frame &f, float const *w, int const *decode_step) {
  HostShared volatile *h = g_ct.host;
  float o[kObs];
  env_observe(f.course, f.course.y, f.course.vy, o);
  bool const flap = forward(w, o).logit > 0.f;
  bool dead;
  float final_score;
  env_step(f.course, flap, dead, final_score);
  f.tick++;
  f.main_tilt = bird_tilt(f.course.vy);
  if (dead) {
    h->game_deaths = h->game_deaths + 1;
    h->game_best = max(h->game_best, int(display_count(final_score)));
  }
  h->game_score = int(display_count(f.course.score));
  f.set_score(f.course.score);

  if (decode_step) {
    h->decode_step = ld_volatile(decode_step);
  }
}

// One frame: tick, render into the back buffer, publish. Called by every
// thread of the block.
__device__ void showcase_frame(GameSmem &s, int const *decode_step) {
  Frame &f = s.frame;
  HostShared volatile *h = g_ct.host;
  load_weights(s.w);
  __syncthreads();
  if (threadIdx.x == 0) {
    showcase_tick(f, s.w, decode_step);
  }
  __syncthreads(); // thread 0's frame state, before everyone draws it
  render_frame(f, const_cast<uint32_t *>(&h->fb[f.buf][0][0]));
  if (threadIdx.x == 0) {
    f.seq++;
    h->frame_buf = f.buf;
    h->frame_time_ns = globaltimer_ns();
    __threadfence_system();
    h->frame_seq = f.seq;
    f.buf ^= 1;
  }
  __syncthreads();
}

// The showcase block: one frame per tick until the co-tenants stop. The
// course carries over to the next launch.
__device__ void run_showcase(int const *decode_step, char *smem) {
  GameSmem &s = *reinterpret_cast<GameSmem *>(smem);
  if (threadIdx.x == 0) {
    Frame &f = s.frame;
    f.course = *g_ct.showcase_env;
    f.tick = 0;
    f.buf = 0;
    f.seq = g_ct.host->frame_seq;
    f.show_main = 1;
    f.show_ai = 0;
  }
  long long next_tick = resume_ticks();
  while (wait_tick(next_tick, s)) {
    showcase_frame(s, decode_step);
  }
  save_ticks(next_tick);
  if (threadIdx.x == 0) {
    *g_ct.showcase_env = s.frame.course;
  }
}

} // namespace cotenant
