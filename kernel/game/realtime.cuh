// SPDX-License-Identifier: Apache-2.0
// The game block (showcase or arcade) runs in real time: one tick every
// kTickNs of GPU time until the co-tenants are told to stop.
#pragma once

#include "../core/device_util.cuh"
#include "../core/state.cuh"
#include "frame.cuh"

namespace cotenant {

struct GameSmem {
  float w[kParamsPadded]; // the policy weights for this tick
  Frame frame;
  int keep_running; // wait_tick's decision, for every thread
  int slot_active;  // arcade: whether the slot being drawn is in use
};

// The tick schedule carries over from one launch to the next. Mirage relaunches
// the kernel for every LLM request, several times a second; restarting the
// schedule at each launch would fire a tick early, a visible hitch.
__device__ long long resume_ticks() {
  return ld_volatile(g_ct.next_tick_ns);
}

// Called by every thread when the game block stops; thread 0 owns the
// schedule.
__device__ void save_ticks(long long next_tick) {
  if (threadIdx.x == 0) {
    st_volatile(g_ct.next_tick_ns, next_tick);
  }
}

// Waits until the next tick and returns true, or returns false once a stop is
// requested. Only thread 0 reads the clock and the stop flags; it broadcasts
// its decision so every thread takes the same branch.
__device__ bool wait_tick(long long &next_tick, GameSmem &s) {
  constexpr long long kMaxLagNs = 10 * kTickNs; // beyond this, skip ahead
  constexpr unsigned kPollNs = 200000;
  if (threadIdx.x == 0) {
    for (;;) {
      if (ld_volatile(g_ct.stop) | ld_volatile(&g_ct.host->stop)) {
        s.keep_running = 0;
        break;
      }
      long long const now = globaltimer_ns();
      if (now >= next_tick) {
        // Fell far behind (e.g. a relaunch): resynchronize instead of racing.
        next_tick =
            now - next_tick > kMaxLagNs ? now + kTickNs : next_tick + kTickNs;
        s.keep_running = 1;
        break;
      }
      __nanosleep(kPollNs);
    }
  }
  __syncthreads();
  return s.keep_running != 0;
}

} // namespace cotenant
