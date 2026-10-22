// SPDX-License-Identifier: Apache-2.0
// Host side: allocates and initializes the co-tenants' state, maps the shared
// block (pinned memory or POSIX shared memory), shares the trainer's buffers
// with a game in another process over CUDA IPC, and records the single-game
// view to video.
//
// Entry points used by Mirage: host_init() once, then host_start() before and
// host_stop() after each launch; host_shared()->stop stops the co-tenants.
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../core/state.cuh"
#include "../game/physics.cuh"

#define CT_CUDA(call)                                                          \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "cotenant %s:%d %s\n",                                           \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace cotenant {

struct Host {
  State state{}; // what g_ct points to
  HostShared *shared = nullptr;
  cudaStream_t reset_stream = nullptr;
  std::thread recorder;
  std::atomic<bool> recording{false};
  int recordings = 0; // numbers the output files of each launch
  bool ready = false;
};

inline Host &host() {
  static Host h;
  return h;
}

inline HostShared *host_shared() {
  return host().shared;
}

// ---------------------------------------------------------------- helpers

[[noreturn]] inline void fail(char const *what, std::string const &detail) {
  fprintf(stderr, "cotenant: %s %s\n", what, detail.c_str());
  exit(1);
}

inline long long env_or(char const *name, long long fallback) {
  char const *v = getenv(name);
  return v ? atoll(v) : fallback;
}

// Zero-filled device memory for n objects.
template <typename T> T *dev_alloc(size_t n = 1) {
  T *p;
  CT_CUDA(cudaMalloc(&p, n * sizeof(T)));
  CT_CUDA(cudaMemset(p, 0, n * sizeof(T)));
  return p;
}

template <typename T> void upload(T *dst, T const *src, size_t n = 1) {
  CT_CUDA(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------- initial
// state

// A fresh course drawn from `seed`.
inline Env make_env(uint32_t seed) {
  Env e{};
  e.rng = seed;
  env_reset(e);
  e.rng |= 1u; // xorshift must never reach 0
  return e;
}

// Seed of learner environment i. Index num_learners * kEnvsPerLearner (one
// past the last learner) seeds the showcase game.
inline uint32_t env_seed(size_t i) {
  uint32_t r = 0x12345u + uint32_t(i) * 7919u;
  for (int k = 0; k < 4; k++) {
    xorshift32(r);
  }
  return r;
}

// The showcase game of a separate game process starts on the same course as
// that of a 32-block co-tenant run (31 learners).
constexpr size_t kSeparateGameSeedIndex = size_t(31) * kEnvsPerLearner;

// PyTorch's nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
inline std::vector<float> initial_weights() {
  std::vector<float> w(kParamsPadded, 0.f);
  uint32_t rng = 0x9E3779B9u;
  auto fill = [&](int offset, int count, int fan_in) {
    float const bound = 1.f / std::sqrt(float(fan_in));
    for (int i = 0; i < count; i++) {
      w[offset + i] = (uniform01(rng) * 2.f - 1.f) * bound;
    }
  };
  fill(kOffW1, kHidden * kObs, kObs);
  fill(kOffB1, kHidden, kObs);
  fill(kOffW2, kHidden * kHidden, kHidden);
  fill(kOffB2, kHidden, kHidden);
  fill(kOffW3, kHeads * kHidden, kHidden);
  fill(kOffB3, kHeads, kHidden);
  return w;
}

// The host block lives in pinned memory mapped into the kernel. With
// $COTENANT_SHM it is a POSIX shared-memory segment instead, so a trainer and
// a game in separate processes share counters (the creator must start from a
// fresh segment, which is zero-filled).
inline void map_host_shared(Host &h) {
  if (char const *shm = getenv("COTENANT_SHM")) {
    int fd = shm_open(shm, O_CREAT | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, sizeof(HostShared)) != 0) {
      fail("cannot map shared memory", shm);
    }
    void *p = mmap(
        nullptr, sizeof(HostShared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    CT_CUDA(cudaHostRegister(p, sizeof(HostShared), cudaHostRegisterMapped));
    h.shared = static_cast<HostShared *>(p);
  } else {
    CT_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&h.shared),
                          sizeof(HostShared),
                          cudaHostAllocMapped));
    memset(h.shared, 0, sizeof(HostShared));
  }
  CT_CUDA(cudaHostGetDevicePointer(
      reinterpret_cast<void **>(&h.state.host), h.shared, 0));
}

inline void publish_state(Host &h) {
  map_host_shared(h);
  CT_CUDA(cudaMemcpyToSymbol(g_ct, &h.state, sizeof(State)));
  h.ready = true;
}

// Allocates and initializes all co-tenant state for `num_blocks` co-tenant
// blocks. Runs once, before the first launch; the state persists across
// launches.
inline void host_init(int num_blocks) {
  Host &h = host();
  if (h.ready) {
    return;
  }
  State &s = h.state;
  int const nl = num_blocks - 1;
  size_t const num_envs = size_t(nl) * kEnvsPerLearner;
  s.mode = Mode(env_or("COTENANT_MODE", kModeShowcase));
  s.num_learners = nl;
  s.max_updates = env_or("COTENANT_MAX_UPDATES", 1LL << 62);
  s.stop = dev_alloc<int>();

  std::vector<float> const w = initial_weights();
  s.weights = dev_alloc<float>(kParamsPadded);
  upload(s.weights, w.data(), w.size());
  s.adam_m = dev_alloc<float>(kParamsPadded);
  s.adam_v = dev_alloc<float>(kParamsPadded);
  s.progress = dev_alloc<Progress>();

  std::vector<Env> envs(num_envs);
  for (size_t i = 0; i < num_envs; i++) {
    envs[i] = make_env(env_seed(i));
  }
  s.learner_envs = dev_alloc<Env>(num_envs);
  upload(s.learner_envs, envs.data(), num_envs);
  s.rollouts = dev_alloc<Rollout>(nl);
  s.partial_grads = dev_alloc<float>(size_t(nl) * kParamsPadded);
  s.episode_stats = dev_alloc<EpisodeStats>(nl);
  s.barrier = dev_alloc<LearnerBarrier>();

  Env const showcase = make_env(env_seed(num_envs));
  s.showcase_env = dev_alloc<Env>();
  upload(s.showcase_env, &showcase);
  std::vector<SlotState> slots(kSlots); // all kIdle
  for (int i = 0; i < kSlots; i++) {
    slots[i].course = make_env(0xC0FFEEu + 977u * i);
  }
  s.slots = dev_alloc<SlotState>(kSlots);
  upload(s.slots, slots.data(), kSlots);
  s.next_tick_ns = dev_alloc<long long>();

  publish_state(h);
}

// ---------------------------------------------------------------- CUDA IPC
// A trainer process publishes the buffers the game reads; a game process
// (tools/separate.cu) opens them.

struct IpcHandles {
  cudaIpcMemHandle_t weights, progress;
};

inline void export_ipc(std::string const &path) {
  State const &s = host().state;
  IpcHandles handles;
  CT_CUDA(cudaIpcGetMemHandle(&handles.weights, s.weights));
  CT_CUDA(cudaIpcGetMemHandle(&handles.progress, s.progress));
  // Written under another name and renamed, so a reader never sees half.
  std::string const tmp = path + ".tmp";
  FILE *f = fopen(tmp.c_str(), "wb");
  if (!f || fwrite(&handles, sizeof(handles), 1, f) != 1) {
    fail("cannot write", tmp);
  }
  fclose(f);
  rename(tmp.c_str(), path.c_str());
}

// Sets up a game-only process on the buffers of the trainer that wrote `path`.
inline void import_ipc_for_game(std::string const &path) {
  Host &h = host();
  State &s = h.state;
  IpcHandles handles;
  FILE *f = fopen(path.c_str(), "rb");
  if (!f || fread(&handles, sizeof(handles), 1, f) != 1) {
    fail("cannot read", path);
  }
  fclose(f);
  auto open = [](cudaIpcMemHandle_t handle) {
    void *p;
    CT_CUDA(cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess));
    return p;
  };
  s.weights = static_cast<float *>(open(handles.weights));
  s.progress = static_cast<Progress *>(open(handles.progress));
  s.stop = dev_alloc<int>();
  Env const showcase = make_env(env_seed(kSeparateGameSeedIndex));
  s.showcase_env = dev_alloc<Env>();
  upload(s.showcase_env, &showcase);
  s.next_tick_ns = dev_alloc<long long>();
  publish_state(h);
}

// ---------------------------------------------------------------- recording

// Quotes a path for /bin/sh, so any $COTENANT_OUT works in the ffmpeg command.
inline std::string shell_quote(std::string const &s) {
  std::string quoted = "'";
  for (char c : s) {
    quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
  }
  return quoted + "'";
}

// Streams the single-game view to <dir>/game_<id>.mp4 and logs the live
// counters while the kernel runs, until h.recording is cleared.
//   frames_<id>.csv  one row per rendered frame
//   video_<id>.csv   one row per video frame; the video is paced by the wall
//                    clock and repeats the newest frame, so a stalled game
//                    shows as a frozen picture
inline void record_frames(Host &h, std::string const &dir, int id) {
  using Clock = std::chrono::steady_clock;
  std::string const n = std::to_string(id);
  std::string const ffmpeg =
      "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s " +
      std::to_string(kScreenW) + "x" + std::to_string(kScreenH) + " -r " +
      std::to_string(kTicksPerSecond) +
      " -i - -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p " +
      shell_quote(dir + "/game_" + n + ".mp4");
  FILE *frame_log = fopen((dir + "/frames_" + n + ".csv").c_str(), "w");
  FILE *video_log = fopen((dir + "/video_" + n + ".csv").c_str(), "w");
  if (!frame_log || !video_log) {
    // Not fatal: this thread cannot exit the process while the kernel runs.
    fprintf(stderr,
            "cotenant: cannot write logs in %s, not recording\n",
            dir.c_str());
    if (frame_log) {
      fclose(frame_log);
    }
    if (video_log) {
      fclose(video_log);
    }
    return;
  }
  FILE *video = popen(ffmpeg.c_str(), "w");
  char const *header =
      "frame,host_ns,wall_ns,gpu_ns,decode_step,updates,episodes,train_mean,"
      "train_best,game_score,game_best,game_deaths\n";
  fputs(header, frame_log);
  fputs(header, video_log);

  HostShared volatile *sh = h.shared;
  auto const t0 = Clock::now();
  auto log_row = [&](FILE *f, long long seq) {
    long long const host_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
            .count();
    long long const wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    fprintf(f,
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.3f,%.1f,%d,%d,%d\n",
            seq,
            host_ns,
            wall_ns,
            sh->frame_time_ns,
            sh->decode_step,
            sh->updates,
            sh->episodes,
            sh->train_mean_score,
            sh->train_best_score,
            sh->game_score,
            sh->game_best,
            sh->game_deaths);
  };

  std::vector<uint32_t> frame(kScreenW * kScreenH, 0xff000000u);
  long long last_seq = sh->frame_seq;
  auto next_video_frame = t0;
  while (h.recording.load()) {
    long long const seq = sh->frame_seq;
    if (seq != last_seq) {
      last_seq = seq;
      memcpy(frame.data(),
             const_cast<uint32_t *>(&sh->fb[sh->frame_buf][0][0]),
             frame.size() * sizeof(uint32_t));
      log_row(frame_log, seq);
    }
    if (Clock::now() >= next_video_frame) {
      if (video) {
        fwrite(frame.data(), sizeof(uint32_t), frame.size(), video);
      }
      log_row(video_log, last_seq);
      next_video_frame += std::chrono::nanoseconds(kTickNs);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  if (video) {
    pclose(video);
  }
  fclose(frame_log);
  fclose(video_log);
}

// ---------------------------------------------------------------- launches

// Before each launch: clears the stop flags and, with $COTENANT_OUT set,
// starts recording the single-game view into that directory.
inline void host_start() {
  Host &h = host();
  // On its own stream so that it completes before any co-tenant block of the
  // next launch starts, whatever streams the launch uses.
  if (h.reset_stream == nullptr) {
    CT_CUDA(cudaStreamCreateWithFlags(&h.reset_stream, cudaStreamNonBlocking));
  }
  CT_CUDA(cudaMemsetAsync(h.state.stop, 0, sizeof(int), h.reset_stream));
  CT_CUDA(cudaStreamSynchronize(h.reset_stream));
  h.shared->stop = 0;
  char const *out = getenv("COTENANT_OUT");
  if (out == nullptr) {
    return;
  }
  h.recording = true;
  h.recorder =
      std::thread(record_frames, std::ref(h), std::string(out), h.recordings++);
}

// After each launch: stops the recording and prints the counters (unless
// $COTENANT_QUIET is set).
inline void host_stop() {
  Host &h = host();
  if (h.recording.exchange(false)) {
    h.recorder.join();
  }
  if (getenv("COTENANT_QUIET") != nullptr) {
    return;
  }
  HostShared const *sh = h.shared;
  printf(
      "[cotenant] updates %lld episodes %lld train_mean %.2f train_best %.0f "
      "game_best %d game_deaths %d frames %lld\n",
      sh->updates,
      sh->episodes,
      sh->train_mean_score,
      sh->train_best_score,
      sh->game_best,
      sh->game_deaths,
      sh->frame_seq);
}

} // namespace cotenant
