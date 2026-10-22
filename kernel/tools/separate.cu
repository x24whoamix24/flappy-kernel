// SPDX-License-Identifier: Apache-2.0
// The co-tenant workloads as ordinary separate GPU processes, for comparison
// with the megakernel:
//   separate train <blocks> <ipc_file> <stop_file>
//       persistent A2C learner kernel ($COTENANT_MODE=3); publishes the
//       weights over CUDA IPC and logs updates to stdout every 250 ms
//   separate game <ipc_file> <stop_file>
//       30 Hz game loop launching one tick+render kernel per frame on the
//       trainer's weights; frames and video go to $COTENANT_OUT
// Both run until <stop_file> exists. Share counters with $COTENANT_SHM.
#include "launch.cuh"

using namespace cotenant;

static_assert(sizeof(Frame) % sizeof(int) == 0, "Frame is copied as ints");

// One showcase frame. The frame lives in global memory between launches and
// in shared memory while the kernel runs.
__global__ void __launch_bounds__(kThreads, 1)
    showcase_frame_kernel(Frame *frame) {
  extern __shared__ __align__(16) char smem[];
  GameSmem &s = *reinterpret_cast<GameSmem *>(smem);
  constexpr int kWords = int(sizeof(Frame) / sizeof(int));
  int *const local = reinterpret_cast<int *>(&s.frame);
  int *const global = reinterpret_cast<int *>(frame);
  for (int i = threadIdx.x; i < kWords; i += kThreads) {
    local[i] = global[i];
  }
  __syncthreads();
  showcase_frame(s, nullptr);
  for (int i = threadIdx.x; i < kWords; i += kThreads) {
    global[i] = local[i];
  }
}

static bool exists(std::string const &path) {
  return access(path.c_str(), F_OK) == 0;
}

static long long wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static void print_progress(HostShared const volatile *sh) {
  printf("%lld,%lld,%lld,%.3f\n",
         wall_ns(),
         sh->updates,
         sh->episodes,
         sh->train_mean_score);
  fflush(stdout);
}

static int
cmd_train(int blocks, std::string const &ipc, std::string const &stop) {
  host_init(blocks);
  export_ipc(ipc);
  cudaStream_t stream;
  CT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  launch_cotenants(blocks, stream);
  HostShared volatile *sh = host_shared();
  printf("wall_ns,updates,episodes,train_mean\n");
  while (!exists(stop)) {
    print_progress(sh);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  stop_cotenants();
  print_progress(sh);
  return 0;
}

// A conventional game loop: one tick per kTickNs deadline; when the GPU runs
// late, missed ticks are dropped rather than replayed.
static int cmd_game(std::string const &ipc, std::string const &stop) {
  while (!exists(ipc)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  import_ipc_for_game(ipc);
  CT_CUDA(cudaFuncSetAttribute(showcase_frame_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kCotenantSmemBytes));
  Frame initial{};
  initial.show_main = 1;
  CT_CUDA(cudaMemcpy(&initial.course,
                     host().state.showcase_env,
                     sizeof(Env),
                     cudaMemcpyDeviceToHost));
  Frame *frame = dev_alloc<Frame>();
  upload(frame, &initial);
  cudaStream_t stream;
  CT_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  host_start();

  auto next = std::chrono::steady_clock::now();
  long long launched = 0, dropped = 0;
  while (!exists(stop)) {
    std::this_thread::sleep_until(next);
    showcase_frame_kernel<<<1, kThreads, kCotenantSmemBytes, stream>>>(frame);
    CT_CUDA(cudaStreamSynchronize(stream));
    launched++;
    next += std::chrono::nanoseconds(kTickNs);
    auto const now = std::chrono::steady_clock::now();
    while (next < now) {
      next += std::chrono::nanoseconds(kTickNs);
      dropped++;
    }
  }
  host_stop();
  printf("[game] frames %lld dropped_ticks %lld\n", launched, dropped);
  return 0;
}

int main(int argc, char **argv) {
  std::string const cmd = argc > 1 ? argv[1] : "";
  if (cmd == "train" && argc > 4) {
    return cmd_train(atoi(argv[2]), argv[3], argv[4]);
  }
  if (cmd == "game" && argc > 3) {
    return cmd_game(argv[2], argv[3]);
  }
  fprintf(stderr,
          "usage: separate train <blocks> <ipc_file> <stop_file> | "
          "game <ipc_file> <stop_file>\n");
  return 1;
}
