// SPDX-License-Identifier: Apache-2.0
// The co-tenant workloads without the LLM, for testing and measurement.
//   standalone run [blocks] [secs]    game + training, printing rates
//   standalone grad <dir>             gradient of a fixed batch, for
//                                     reference/check_grad.py
//   standalone probe <blocks>         HBM bandwidth on the other SMs, alone
//                                     and next to training
//   standalone bounded <blocks>       training only, until
//                                     $COTENANT_MAX_UPDATES (for ncu)
//   standalone export <blocks> <dir>  trains, saving policy snapshots as JSON
//   standalone layout                 byte offsets of the shared block, for
//                                     server/
#include <cstddef>

#include "launch.cuh"

using namespace cotenant;

using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- grad

// The gradient of `n` samples (kObs observations, action, return each),
// summed like the trainer does.
__global__ void
grad_kernel(float const *w_in, float const *samples, int n, float *g_out) {
  extern __shared__ __align__(16) char smem[];
  float *w = reinterpret_cast<float *>(smem);
  float *g = w + kParamsPadded;
  for (int p = threadIdx.x; p < kParamsPadded; p += blockDim.x) {
    w[p] = w_in[p];
    g[p] = 0.f;
  }
  __syncthreads();
  for (int s = threadIdx.x; s < n; s += blockDim.x) {
    float const *x = samples + s * (kObs + 2);
    accumulate_grad(w, g, x, x[kObs], x[kObs + 1], threadIdx.x % kWarpSize);
  }
  __syncthreads();
  for (int p = threadIdx.x; p < kParamsPadded; p += blockDim.x) {
    g_out[p] = g[p];
  }
}

static std::vector<float> read_floats(std::string const &path) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) {
    fail("cannot open", path);
  }
  fseek(f, 0, SEEK_END);
  long const bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<float> v(bytes / sizeof(float));
  if (fread(v.data(), sizeof(float), v.size(), f) != v.size()) {
    fail("cannot read", path);
  }
  fclose(f);
  return v;
}

// Reads <dir>/w.bin and <dir>/samples.bin, writes <dir>/grad.bin.
static int cmd_grad(std::string const &dir) {
  std::vector<float> const w = read_floats(dir + "/w.bin");
  std::vector<float> const samples = read_floats(dir + "/samples.bin");
  int const n = int(samples.size() / (kObs + 2));
  // accumulate_grad's shuffles need every lane of a warp to have a sample.
  if (n % kWarpSize != 0) {
    fail("the sample count must be a multiple of 32, not", std::to_string(n));
  }
  float *d_w = dev_alloc<float>(kParamsPadded);
  float *d_samples = dev_alloc<float>(samples.size());
  float *d_g = dev_alloc<float>(kParamsPadded);
  upload(d_w, w.data(), kParamsPadded);
  upload(d_samples, samples.data(), samples.size());
  grad_kernel<<<1, kThreads, 2 * kParamsPadded * sizeof(float)>>>(
      d_w, d_samples, n, d_g);
  CT_CUDA(cudaDeviceSynchronize());
  std::vector<float> g(kParamsPadded);
  CT_CUDA(cudaMemcpy(
      g.data(), d_g, kParamsPadded * sizeof(float), cudaMemcpyDeviceToHost));
  std::string const out = dir + "/grad.bin";
  FILE *f = fopen(out.c_str(), "wb");
  if (!f || fwrite(g.data(), sizeof(float), g.size(), f) != g.size()) {
    fail("cannot write", out);
  }
  fclose(f);
  printf("wrote %d grads for %d samples\n", kParamsPadded, n);
  return 0;
}

// ---------------------------------------------------------------- probe

// Streams a large buffer the way memory-bound decode streams weights.
__global__ void
read_bw_kernel(int4 const *buf, size_t n, int iters, int *sink) {
  int acc = 0;
  for (int it = 0; it < iters; it++) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
      int4 const v = __ldcs(buf + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x7fffffff) {
    *sink = acc; // keeps the loads alive
  }
}

// Read bandwidth of `blocks` SMs over 4 passes of an 8 GiB buffer, in GB/s.
static double probe_bandwidth(int blocks, cudaStream_t stream) {
  constexpr size_t kBytes = size_t(8) << 30;
  constexpr int kIters = 4;
  static int4 *buf = nullptr;
  static int *sink = nullptr;
  if (!buf) {
    CT_CUDA(cudaMalloc(&buf, kBytes));
    CT_CUDA(cudaMemset(buf, 1, kBytes));
    CT_CUDA(cudaMalloc(&sink, sizeof(int)));
  }
  cudaEvent_t start, end;
  CT_CUDA(cudaEventCreate(&start));
  CT_CUDA(cudaEventCreate(&end));
  CT_CUDA(cudaEventRecord(start, stream));
  read_bw_kernel<<<blocks, 1024, 0, stream>>>(
      buf, kBytes / sizeof(int4), kIters, sink);
  CT_CUDA(cudaEventRecord(end, stream));
  CT_CUDA(cudaEventSynchronize(end));
  float ms;
  CT_CUDA(cudaEventElapsedTime(&ms, start, end));
  return double(kBytes) * kIters / (ms * 1e-3) / 1e9;
}

static int cmd_probe(int blocks) {
  int device_sms;
  CT_CUDA(
      cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, 0));
  int const sms = device_sms - blocks;
  cudaStream_t stream;
  CT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  probe_bandwidth(sms, stream); // warm-up
  printf("alone:          %.0f GB/s on %d SMs\n",
         probe_bandwidth(sms, stream),
         sms);

  host_init(blocks);
  host_start();
  launch_cotenants(blocks);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  long long const updates0 = host_shared()->updates;
  auto const t0 = Clock::now();
  double const bandwidth = probe_bandwidth(sms, stream);
  double const elapsed = seconds_since(t0);
  long long const updates1 = host_shared()->updates;
  printf("with co-tenant: %.0f GB/s on %d SMs (co-tenant %.0f updates/s)\n",
         bandwidth,
         sms,
         (updates1 - updates0) / elapsed);
  stop_cotenants();
  host_stop();
  return 0;
}

// ---------------------------------------------------------------- layout

// Byte offsets of the shared host block, for processes that map it
// (server/app.py).
static int cmd_layout() {
  printf("{\"size\": %zu, \"kW\": %d, \"kH\": %d, \"slots\": %d, "
         "\"llm_max_tokens\": %d,\n",
         sizeof(HostShared),
         kScreenW,
         kScreenH,
         kSlots,
         kLlmMaxTokens);
  printf(" \"slot\": %zu, \"slot_size\": %zu, \"slot_fb\": %zu,\n",
         offsetof(HostShared, slot),
         sizeof(SlotIO),
         offsetof(HostShared, slot_fb));
  printf(" \"slot_fields\": {\"assigned\": %zu, \"mode\": %zu, "
         "\"start_seq\": %zu, \"flap_seq\": %zu, "
         "\"phase\": %zu, \"you_score\": %zu, \"ai_score\": %zu, "
         "\"you_alive\": %zu, \"ai_alive\": %zu, "
         "\"frame_buf\": %zu, \"frame_seq\": %zu},\n",
         offsetof(SlotIO, assigned),
         offsetof(SlotIO, mode),
         offsetof(SlotIO, start_seq),
         offsetof(SlotIO, flap_seq),
         offsetof(SlotIO, phase),
         offsetof(SlotIO, you_score),
         offsetof(SlotIO, ai_score),
         offsetof(SlotIO, you_alive),
         offsetof(SlotIO, ai_alive),
         offsetof(SlotIO, frame_buf),
         offsetof(SlotIO, frame_seq));
  printf(" \"updates\": %zu, \"episodes\": %zu, \"train_mean_score\": %zu, "
         "\"train_best_score\": %zu,\n",
         offsetof(HostShared, updates),
         offsetof(HostShared, episodes),
         offsetof(HostShared, train_mean_score),
         offsetof(HostShared, train_best_score));
  printf(" \"llm_launch\": %zu, \"llm_prompt_len\": %zu, \"llm_pos\": %zu, "
         "\"llm_tokens\": %zu}\n",
         offsetof(HostShared, llm_launch),
         offsetof(HostShared, llm_prompt_len),
         offsetof(HostShared, llm_pos),
         offsetof(HostShared, llm_tokens));
  return 0;
}

// ---------------------------------------------------------------- export

// Trains and saves the policy weights as JSON once training passes each
// update count in kMarks: <dir>/policy_<mark>.json.
static int cmd_export(int blocks, std::string const &dir) {
  constexpr long long kMarks[] = {0, 300, 600, 800, 1000, 1200, 1500, 4000};
  host_init(blocks);
  cudaStream_t copy_stream;
  CT_CUDA(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  host_start();
  launch_cotenants(blocks);
  HostShared volatile *sh = host_shared();
  std::vector<float> w(kParamsPadded);
  for (long long mark : kMarks) {
    while (sh->updates < mark) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    long long const updates = sh->updates;
    CT_CUDA(cudaMemcpyAsync(w.data(),
                            host().state.weights,
                            kParamsPadded * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            copy_stream));
    CT_CUDA(cudaStreamSynchronize(copy_stream));
    std::string const path = dir + "/policy_" + std::to_string(mark) + ".json";
    FILE *f = fopen(path.c_str(), "w");
    if (!f) {
      fail("cannot write", path);
    }
    fprintf(f,
            "{\"updates\": %lld, \"train_mean\": %.3f, \"w\": [",
            updates,
            sh->train_mean_score);
    for (int p = 0; p < kParams; p++) {
      fprintf(f, "%s%.6g", p ? "," : "", w[p]);
    }
    fprintf(f, "]}\n");
    fclose(f);
    printf("saved %s (updates %lld, mean %.2f)\n",
           path.c_str(),
           updates,
           sh->train_mean_score);
    fflush(stdout);
  }
  stop_cotenants();
  host_stop();
  return 0;
}

// ---------------------------------------------------------------- bounded

static int cmd_bounded(int blocks) {
  setenv("COTENANT_MODE", "3", 1); // training only: it stops by itself
  host_init(blocks);
  launch_cotenants(blocks);
  CT_CUDA(cudaDeviceSynchronize());
  printf("updates %lld\n", host_shared()->updates);
  return 0;
}

// ---------------------------------------------------------------- run

static int cmd_run(int blocks, double secs) {
  host_init(blocks);
  host_start();
  HostShared volatile *sh = host_shared();
  auto const t0 = Clock::now();
  launch_cotenants(blocks);
  long long last_updates = 0;
  for (int s = 1; s <= int(secs); s++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    long long const updates = sh->updates;
    printf("t=%2ds updates %lld (+%lld/s) eps %lld mean %.2f best %.0f "
           "game %d/%d deaths %d frames %lld\n",
           s,
           updates,
           updates - last_updates,
           sh->episodes,
           sh->train_mean_score,
           sh->train_best_score,
           sh->game_score,
           sh->game_best,
           sh->game_deaths,
           sh->frame_seq);
    fflush(stdout);
    last_updates = updates;
  }
  stop_cotenants();
  double const elapsed = seconds_since(t0);
  host_stop();
  printf("elapsed %.2fs, %.1f updates/s, %.1f frames/s\n",
         elapsed,
         sh->updates / elapsed,
         sh->frame_seq / elapsed);
  return 0;
}

// ---------------------------------------------------------------- main

static int usage() {
  fprintf(stderr,
          "usage: standalone run [blocks] [secs] | grad <dir> | "
          "probe <blocks> | bounded <blocks> | export <blocks> <dir> | "
          "layout\n");
  return 2;
}

int main(int argc, char **argv) {
  std::string const cmd = argc > 1 ? argv[1] : "run";
  if (cmd == "run") {
    return cmd_run(argc > 2 ? atoi(argv[2]) : 32,
                   argc > 3 ? atof(argv[3]) : 10.0);
  }
  if (cmd == "layout") {
    return cmd_layout();
  }
  if (cmd == "grad" && argc > 2) {
    return cmd_grad(argv[2]);
  }
  if (cmd == "probe" && argc > 2) {
    return cmd_probe(atoi(argv[2]));
  }
  if (cmd == "bounded" && argc > 2) {
    return cmd_bounded(atoi(argv[2]));
  }
  if (cmd == "export" && argc > 3) {
    return cmd_export(atoi(argv[2]), argv[3]);
  }
  return usage();
}
