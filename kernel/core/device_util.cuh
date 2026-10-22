// SPDX-License-Identifier: Apache-2.0
// Small helpers: the xorshift random generator (shared with the host so both
// sides draw the same courses), the GPU clock, memory accesses for values
// that other blocks or the host change while a block runs, and warp
// reductions.
#pragma once

#include <cstdint>

#include "config.h"

namespace cotenant {

// ---------------------------------------------------------------- random

__host__ __device__ __forceinline__ uint32_t xorshift32(uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

// Uniform in [0, 1), from the top 24 bits.
__host__ __device__ __forceinline__ float uniform01(uint32_t &s) {
  return (xorshift32(s) >> 8) * (1.f / 16777216.f);
}

// ---------------------------------------------------------------- clock

__device__ __forceinline__ long long globaltimer_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------- memory

__device__ __forceinline__ unsigned ld_acquire(unsigned const *p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ int ld_acquire(int const *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

// Accesses that always go to memory, for values another block or the host
// changes while this one runs.
template <typename T> __device__ __forceinline__ T ld_volatile(T const *p) {
  return *const_cast<T const volatile *>(p);
}

template <typename T> __device__ __forceinline__ void st_volatile(T *p, T v) {
  *const_cast<T volatile *>(p) = v;
}

// ---------------------------------------------------------------- warp

constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = kWarpSize / 2; m > 0; m >>= 1) {
    v += __shfl_xor_sync(kFullWarp, v, m);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = kWarpSize / 2; m > 0; m >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(kFullWarp, v, m));
  }
  return v;
}

} // namespace cotenant
