// SPDX-License-Identifier: Apache-2.0
// The arcade (COTENANT_MODE=4): up to kSlots - 1 players, each racing the AI
// on their own course, plus the watch view. One block, every tick:
//   1. steps every game slot: the player's bird with the flaps the host
//      relays, the AI's bird with the live policy, through the same pipes,
//   2. copies the tokens the LLM decoded since the last tick to the host,
//   3. renders each active slot into the back half of its double buffer and
//      publishes it.
#pragma once

#include "../core/device_util.cuh"
#include "../training/policy.cuh"
#include "physics.cuh"
#include "realtime.cuh"
#include "render.cuh"

namespace cotenant {

// In watch mode the AI starts over this many ticks after it crashes.
constexpr int kWatchRestartTicks = 30;

__device__ void slot_reset(SlotState &s, SlotMode mode) {
  env_reset(s.course);
  s.ai_y = s.course.y;
  s.ai_vy = 0.f;
  s.ai_score = 0.f;
  s.you_alive = mode != kWatch;
  s.ai_alive = mode != kSolo;
  s.ai_dead_ticks = 0;
  s.mode = mode;
  s.phase = mode == kWatch ? kPlaying : kReady;
  s.tick = 0;
}

// Moves the pipes and both birds of a game in progress.
__device__ void slot_play(SlotState &s, float const *w, bool flap) {
  float const passed = move_pipes(s.course);
  s.tick++;
  if (s.you_alive) {
    s.you_alive = fly_bird(s.course, s.course.y, s.course.vy, flap);
    s.course.score += s.you_alive ? passed : 0.f;
  }
  if (s.ai_alive) {
    float o[kObs];
    env_observe(s.course, s.ai_y, s.ai_vy, o);
    bool const ai_flap = forward(w, o).logit > 0.f;
    s.ai_alive = fly_bird(s.course, s.ai_y, s.ai_vy, ai_flap);
    s.ai_score += s.ai_alive ? passed : 0.f;
  } else {
    s.ai_dead_ticks++;
  }
  if (s.mode == kWatch && !s.ai_alive && s.ai_dead_ticks > kWatchRestartTicks) {
    slot_reset(s, kWatch);
  } else if (s.mode != kWatch && !s.you_alive) {
    s.phase = kOver;
  }
}

// One tick of slot i, by thread 0. The host's controls are read once, here,
// and validated; everything else the slot needs lives in device memory. The
// host writes `mode`, then bumps `start_seq`, then sets `assigned`; the
// acquire loads keep the later reads from seeing older values.
__device__ void slot_tick(int i, float const *w) {
  SlotState &s = g_ct.slots[i];
  SlotIO volatile &io = g_ct.host->slot[i];
  if (!ld_acquire_sys(&io.assigned)) {
    s.phase = kIdle;
    io.phase = kIdle;
    return;
  }
  int const start_seq = ld_acquire_sys(&io.start_seq);
  int const flap_seq = io.flap_seq;
  int const requested_mode = io.mode;
  SlotMode const mode = requested_mode == kSolo || requested_mode == kWatch
                            ? SlotMode(requested_mode)
                            : kRace;
  if (start_seq != s.start_seen || s.phase == kIdle) {
    s.start_seen = start_seq;
    s.flap_seen = flap_seq;
    slot_reset(s, mode);
  }
  bool const flap = flap_seq != s.flap_seen;
  s.flap_seen = flap_seq;
  if (s.phase == kReady && flap) {
    s.phase = kPlaying;
  }
  if (s.phase == kPlaying) {
    slot_play(s, w, flap);
  }
  io.phase = s.phase;
  io.you_score = int(display_count(s.course.score));
  io.ai_score = int(display_count(s.ai_score));
  io.you_alive = s.you_alive;
  io.ai_alive = s.ai_alive;
}

// Where Mirage keeps the request it is decoding; all null without an LLM.
struct LlmView {
  int const *decode_step;   // position of the newest token
  int const *prompt_length; // the prompt fills positions [0, prompt_length)
  long long const *tokens;  // the request's token row
  int const *request_id;    // online serving only (see LlmFeed), else null
};

// Copies the LLM's tokens into host memory as they are decoded. Thread 0
// only. Mirage either serves one request per launch (`request_id` null) or,
// online, every request in one launch: then `request_id` holds the id of the
// request in the token row (-1 while the row is free), and the scheduler
// release-stores `decode_step` after writing the tokens up to it. Online, both
// live in host memory, which the host writes too, so they are read at system
// scope; the scheduler sets the prompt length before publishing the id.
struct LlmFeed {
  int request; // the request being copied, -1 for none
  int copied;  // highest position already in host memory

  // Starts copying request `id`. HostShared::llm_launch names the request
  // whose tokens llm_tokens holds.
  __device__ void begin(int id, int len) {
    HostShared volatile *h = g_ct.host;
    request = id;
    copied = len - 1;
    h->llm_prompt_len = len;
    h->llm_pos = copied;
    __threadfence_system();
    h->llm_launch = id;
  }

  // At the start of a launch. Per-launch serving numbers requests by launch.
  __device__ void start(LlmView const &llm) {
    if (llm.request_id != nullptr) {
      request = -1;
      return;
    }
    HostShared volatile *h = g_ct.host;
    begin(int(h->llm_launch) + 1,
          llm.prompt_length ? ld_volatile(llm.prompt_length) : 0);
  }

  // Every tick, and with `flush` once more when the launch ends.
  __device__ void update(LlmView const &llm, bool flush) {
    if (llm.decode_step == nullptr || llm.tokens == nullptr) {
      return;
    }
    if (llm.request_id == nullptr) {
      // The position can become visible before its token: hold the newest
      // one back until the launch ends.
      int const pos = ld_acquire(llm.decode_step);
      copy_upto(llm.tokens, flush ? pos : pos - 1);
      return;
    }
    int const id = ld_acquire_sys(llm.request_id);
    if (id < 0) {
      return;
    }
    if (id != request) {
      begin(id, ld_volatile(llm.prompt_length));
    }
    int const pos =
        ld_acquire_sys(llm.decode_step); // -1 once the host has read it
    if (pos >= 0 && ld_acquire_sys(llm.request_id) == id) {
      copy_upto(llm.tokens, pos);
    }
  }

  __device__ void copy_upto(long long const *tokens, int upto) {
    HostShared volatile *h = g_ct.host;
    upto = min(upto, kLlmMaxTokens - 1);
    for (int t = copied + 1; t <= upto; t++) {
      h->llm_tokens[t] = __ldcg(tokens + t);
    }
    if (upto > copied) {
      copied = upto;
      __threadfence_system();
      h->llm_pos = upto;
    }
  }
};

// Thread 0: fills the frame of an active slot and flips its buffer.
__device__ void slot_frame(Frame &f, SlotState &s) {
  f.course = s.course;
  f.tick = s.tick;
  f.show_main = s.mode != kWatch;
  f.show_ai = s.mode != kSolo;
  f.ai_alive = s.ai_alive;
  f.ai_y = s.ai_y;
  f.main_tilt = s.you_alive ? bird_tilt(s.course.vy) : kCrashedTilt;
  f.ai_tilt = s.ai_alive ? bird_tilt(s.ai_vy) : kCrashedTilt;
  f.set_score(s.mode == kWatch ? s.ai_score : s.course.score);
  s.buf ^= 1; // the buffer index never comes from host memory
  f.buf = s.buf;
}

// The arcade block's main loop.
__device__ void run_arcade(LlmView const &llm, char *smem) {
  GameSmem &s = *reinterpret_cast<GameSmem *>(smem);
  HostShared volatile *h = g_ct.host;
  LlmFeed feed{};
  if (threadIdx.x == 0) {
    feed.start(llm);
  }
  long long next_tick = resume_ticks();
  while (wait_tick(next_tick, s)) {
    load_weights(s.w);
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 0; i < kSlots; i++) {
        slot_tick(i, s.w);
      }
      feed.update(llm, false);
    }
    for (int i = 0; i < kSlots; i++) {
      SlotState &slot = g_ct.slots[i];
      if (threadIdx.x == 0) {
        s.slot_active = slot.phase != kIdle;
        if (s.slot_active) {
          slot_frame(s.frame, slot);
        }
      }
      __syncthreads();
      bool const active = s.slot_active;
      __syncthreads(); // before thread 0 sets slot_active for the next slot
      if (!active) {
        continue;
      }
      render_frame(s.frame,
                   const_cast<uint32_t *>(&h->slot_fb[i][s.frame.buf][0][0]));
      if (threadIdx.x == 0) {
        h->slot[i].frame_buf = slot.buf;
        __threadfence_system();
        h->slot[i].frame_seq = ++slot.frame_seq;
      }
      __syncthreads();
    }
  }
  save_ticks(next_tick);
  if (threadIdx.x == 0) {
    feed.update(llm, true);
  }
}

} // namespace cotenant
