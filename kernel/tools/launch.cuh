// SPDX-License-Identifier: Apache-2.0
// The co-tenant blocks as a kernel of their own, for the tools that run them
// without the LLM.
#pragma once

#include "../cotenant.cuh"

namespace cotenant {

// Dynamic shared memory per co-tenant block, the same as inside Mirage.
constexpr int kCotenantSmemBytes = 200 * 1024;
static_assert(sizeof(LearnerSmem) <= kCotenantSmemBytes &&
                  sizeof(GameSmem) <= kCotenantSmemBytes,
              "a co-tenant block's shared memory does not fit");

__global__ void __launch_bounds__(kThreads, 1) cotenant_kernel() {
  extern __shared__ __align__(16) char smem[];
  run(nullptr, nullptr, nullptr, nullptr, blockIdx.x, smem);
}

// Launches `blocks` co-tenant blocks (block 0 the game, the rest learners).
inline void launch_cotenants(int blocks, cudaStream_t stream = nullptr) {
  static bool configured = false;
  if (!configured) {
    CT_CUDA(cudaFuncSetAttribute(cotenant_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kCotenantSmemBytes));
    configured = true;
  }
  cotenant_kernel<<<blocks, kThreads, kCotenantSmemBytes, stream>>>();
  CT_CUDA(cudaGetLastError());
}

// Asks the running co-tenants to stop and waits for them.
inline void stop_cotenants() {
  host_shared()->stop = 1;
  CT_CUDA(cudaDeviceSynchronize());
}

} // namespace cotenant
